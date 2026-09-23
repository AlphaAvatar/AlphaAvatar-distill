#!/usr/bin/env python3
"""Pre-stage a campaign's verified probes onto a provider network volume.

**Why this exists.** A Phase-C2 continuation must put its campaign's completed
probes on the replacement pod and re-identify them there (resume rule R2/R6).
The probes live on the launcher host; its uplink has measured between 0.32 and
0.91 MB/s, so 22.2 GiB is somewhere between seven and twenty hours. Paying an
L40S to sit through that is the single most expensive way to move bytes this
project has available, and attempt5 died of a storage model that assumed the
transfer away entirely.

A **network volume** removes the transfer from the experiment instead of
budgeting for it. The volume is written once, while nothing expensive is
billing, by the cheapest pod the provider sells. Every later attempt in the
campaign mounts it and the probes are simply *there* — no fetch, no fetch
failure mode, no transport reserve, and no object-store credential anywhere
near the formal pod.

This is ordinary engineering infrastructure, not a scientific attempt. It
decides nothing, it advances no candidate, and it is deliberately filed outside
the campaign's attempt namespace so its cost is not charged to the campaign's
scientific ceiling.

**What it guarantees.**

* At most ONE pod, from ONE create call, registered before anything can reject
  it, and torn down on every exit path with the teardown *verified* by polling
  rather than assumed from a command's exit status.
* A hard spend guard derived from the price the provider actually quoted, not
  from the price that was expected.
* Every staged file is re-hashed **from the bytes on the volume** and compared
  to the launcher host's hash. A transfer that truncated a shard is caught
  here, at CPU-pod prices, rather than on the formal pod.
* Restartable: rsync resumes partial files and a probe whose bytes already
  verify on the volume is skipped, so an interrupted run costs only what it has
  not yet moved.
* Two health gates per draw, because the cheapest pod the provider sells lands
  wherever it has room. The first real staging pod reported a one-minute load
  average of **65 on a two-vCPU share** and decayed to 0.028 MB/s — ten days
  for this job — while the launcher host sat at load 1.1 on sixteen cores.
  A draw is now refused on host load and on a transport floor, before 22 GiB
  is committed to it.

**What it deliberately does not do.** It does not re-identify a probe as a
*measurement* — that is `verify_transferred_leaf` on the formal pod, against
the identity the campaign announced, and moving it here would put the
scientific admission decision in a piece of transport tooling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from aadistill.infrastructure.provider import RunPodProvider  # noqa: E402

SCHEMA = "aadistill.autoinit.c2_probe_volume_staging/v1"

#: The files a probe directory must carry for a continuation to consume it at
#: all. The weights are the measurement, the config identifies them, and the
#: record says WHICH probe they are. A probe's score evidence is not listed
#: here on purpose: `campaign_state` already treats a probe whose rows did not
#: survive as unscored rather than unusable, and that probe legitimately
#: resumes at scoring.
REQUIRED = ("model.safetensors", "config.json", "probe_record.json",
            "durable_ack.json")

#: Where the volume is mounted inside the staging pod, and inside every later
#: pod that attaches it. One spelling, because two processes disagreeing about
#: the mount path is a failure that only shows up once a pod is billing.
MOUNT = "/durable"


class StagingError(RuntimeError):
    """The staging run cannot proceed, or cannot be trusted."""


def say(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


def sha256_file(path: Path, *, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def api_key(config: Path) -> str:
    """The provider key, from the operator's config. Never logged, never stored.

    Read as a value and passed to the provider client; it does not enter the
    evidence record, the transcript or any file this script writes.
    """
    cfg = tomllib.loads(config.read_text())
    key = (cfg.get("apikey") or cfg.get("api_key") or "").strip()
    if not key:
        raise StagingError(f"{config} carries no provider API key")
    return key


# ---------------------------------------------------------------------------
# what is on the launcher host
# ---------------------------------------------------------------------------

def require_local_rsync() -> str:
    """The HOST's half of the transfer contract, checked before a pod exists.

    The rehearsal that found this defect had already verified rsync on the pod
    and then died on the first transfer because the **launcher host** had no
    rsync at all — a two-ended contract with one end checked. The pod was
    billing by then, so a precondition that costs nothing to test was paid for.

    It is deliberately here, in host-only code that runs before `create`,
    rather than beside the pod-side check: the two ends are not one check and
    putting them next to each other is what made it easy to believe they were.
    """
    from shutil import which

    found = which("rsync")
    if not found:
        raise StagingError(
            "rsync is not installed on this launcher host. The transfer needs "
            "it on BOTH ends — resuming a partially sent 2 GiB shard is what "
            "makes a multi-hour upload survivable — and checking only the pod "
            "is how this was discovered from a billing resource.")
    return found


def local_inventory(campaign_root: Path) -> dict[str, dict]:
    """Exactly the probes a continuation could restore, each file hashed.

    **The set comes from `campaign_state`, not from a directory listing**, so
    what is staged is by construction the same set the launcher will later name
    in its continuation manifest. Choosing an attempt directory by hand would
    let the two diverge: `campaign_state` collapses probe ids across attempts
    and refuses probes whose ack or training descriptor is missing, and a
    staging tool that walked one attempt would happily copy bytes the launcher
    then declines to use — or miss ones it needs.

    Hashing ~22 GiB takes a few minutes and is what the destination check
    compares against, so it happens once, before any resource exists.
    """
    from experiments.phase_c2 import behavioural_continuation as BC

    state = BC.campaign_state(campaign_root)
    if not state["probes"]:
        raise StagingError(
            f"{campaign_root} holds no destination-verified probe. There is "
            "nothing for a continuation to restore and nothing to stage.")
    out: dict[str, dict] = {}
    for pid, held in sorted(state["probes"].items()):
        probe = Path(held["durable_path"])
        missing = [n for n in REQUIRED if not (probe / n).is_file()]
        if missing:
            raise StagingError(
                f"{pid}: missing {missing}. A probe without these cannot be "
                "consumed by a continuation, so staging it would move bytes "
                "that a paid pod would then refuse.")
        files: dict[str, dict] = {}
        for f in sorted(probe.rglob("*")):
            if f.is_file():
                rel = str(f.relative_to(probe))
                files[rel] = {"size": f.stat().st_size, "sha256": sha256_file(f)}
        out[pid] = {
            "source_attempt": held["attempt"],
            "durable_path": str(probe),
            "scored": bool(held["scored"]),
            "artifact_digest": (held["identity"] or {}).get("artifact_digest"),
            "files": files,
            "bytes": sum(v["size"] for v in files.values()),
        }
        say(f"  hashed {pid}: {len(files)} files, "
            f"{out[pid]['bytes'] / 2**30:.2f} GiB, "
            f"from {held['attempt']}, scored={held['scored']}")
    if state["rejected"]:
        say(f"  {len(state['rejected'])} probe dir(s) not reusable and not "
            f"staged: {[r['probe_id'] for r in state['rejected']]}")
    return out


# ---------------------------------------------------------------------------
# the pod
# ---------------------------------------------------------------------------

class StagingPod:
    """One CPU pod with the volume attached. Created once, torn down always."""

    def __init__(self, a, provider: RunPodProvider, ev: dict) -> None:
        self.a = a
        self.provider = provider
        self.ev = ev
        self.pod_id = ""
        self.price = 0.0
        self.start_epoch = 0.0
        self.endpoint: tuple[str, str] | None = None

    # -- money -------------------------------------------------------------
    def elapsed_min(self) -> float:
        return 0.0 if not self.start_epoch else (time.time() - self.start_epoch) / 60

    def spent(self) -> float:
        return self.elapsed_min() / 60 * self.price

    def check_budget(self) -> None:
        """Refuse to continue once the quoted rate has spent the allowance.

        Denominated in the price the PROVIDER quoted, not the one that was
        expected, so a pod that came back more expensive than planned shortens
        the window instead of overrunning the envelope invisibly.
        """
        if self.spent() > self.a.max_usd:
            raise StagingError(
                f"staging has spent ${self.spent():.4f} of ${self.a.max_usd:.4f} "
                f"at the quoted ${self.price:.4f}/h; stopping")

    # -- lifecycle ---------------------------------------------------------
    def create(self) -> None:
        """ONE create call. The id is registered before anything may reject it.

        A non-empty id means a resource exists and is billing. Every rejection
        below therefore happens *after* the id is recorded and after teardown is
        possible, because a session that creates a pod and then returns early on
        a price check has still been billed for it.
        """
        deadline = (datetime.now(timezone.utc)
                    + timedelta(minutes=self.a.terminate_after_min))
        cmd = ["runpodctl", "pod", "create",
               "--compute-type", "cpu",
               "--image", self.a.image,
               "--container-disk-in-gb", str(self.a.disk_gb),
               "--network-volume-id", self.a.volume_id,
               "--volume-mount-path", MOUNT,
               "--data-center-ids", self.a.data_center,
               "--ports", "22/tcp",
               "--name", f"aad-{self.a.run_id}",
               "--terminate-after", deadline.strftime("%Y-%m-%dT%H:%M:%SZ")]
        say("create: " + " ".join(shlex.quote(c) for c in cmd))
        raw = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        blob = raw.stdout + raw.stderr
        self.ev["create_raw"] = blob[-4000:]
        pid = ""
        try:
            pid = json.loads(raw.stdout).get("id", "") or ""
        except Exception:                                          # noqa: BLE001
            import re
            m = re.search(r'"id"\s*:\s*"([^"]+)"', blob)
            pid = m.group(1) if m else ""
        if not pid:
            raise StagingError(f"no pod id from the provider: {blob[:600]}")
        self.pod_id = pid
        self.start_epoch = time.time()
        self.ev["pod_id"] = pid
        self.ev["pod_started_utc"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds")
        try:
            quoted = json.loads(raw.stdout).get("costPerHr")
            self.price = float(quoted) if quoted is not None else self.a.price_cap
        except Exception:                                          # noqa: BLE001
            self.price = self.a.price_cap
        self.ev["quoted_usd_per_hour"] = self.price
        say(f"pod {pid} created at ${self.price:.4f}/h")
        if self.price > self.a.price_cap:
            raise StagingError(
                f"provider quoted ${self.price:.4f}/h above the "
                f"${self.a.price_cap:.4f}/h cap")

    def wait_endpoint(self) -> tuple[str, str]:
        """Poll for TCP 22. A failed observation is unknown state, never absence."""
        deadline = time.time() + self.a.startup_limit_min * 60
        transient = 0
        while time.time() < deadline:
            self.check_budget()
            obs = self.provider.observe(
                'query { pod(input:{podId:"%s"}) { runtime { ports '
                '{ ip publicPort privatePort type } } } }' % self.pod_id)
            if not obs.ok:
                transient += 1
                if transient in (1, 10) or transient % 30 == 0:
                    say(f"  control plane silent ({obs.error}) — {transient} so far")
                time.sleep(10)
                continue
            rt = ((obs.data or {}).get("pod") or {}).get("runtime")
            for p in (rt or {}).get("ports") or []:
                if p.get("privatePort") == 22 and p.get("type") == "tcp":
                    ep = (str(p["ip"]), str(p["publicPort"]))
                    self.endpoint = ep
                    say(f"TCP 22 at {ep[0]}:{ep[1]} after "
                        f"{self.elapsed_min():.1f} min")
                    return ep
            time.sleep(10)
        raise StagingError(
            f"no SSH endpoint within {self.a.startup_limit_min} min; the pod "
            "is billing and will be torn down")

    # -- remote execution --------------------------------------------------
    def ssh_base(self) -> list[str]:
        host, port = self.endpoint            # type: ignore[misc]
        return ["ssh", "-p", port, "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
                "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=10",
                f"root@{host}"]

    def run(self, command: str, *, timeout: float = 300) -> subprocess.CompletedProcess:
        return subprocess.run(self.ssh_base() + [command],
                              capture_output=True, text=True, timeout=timeout)

    def wait_ssh(self) -> None:
        for _ in range(40):
            self.check_budget()
            if self.run("true", timeout=30).returncode == 0:
                say(f"ssh reachable — ${self.spent():.4f} spent")
                return
            time.sleep(10)
        raise StagingError("the pod never became reachable over SSH")

    def require_healthy_host(self) -> dict:
        """Record the host's contention. A BACKSTOP, not the instrument.

        This began as a refusal at 8.0 load per vCPU, written after the first
        staging pod decayed from 0.32 MB/s to 0.028 — ten days' worth — while
        reporting a one-minute load of 65 on a two-vCPU share. The launcher
        host was idle at the time (load 1.1 on sixteen cores, rsync using 3.3%
        of one core), so the pod's host was the only candidate.
        
        The next draw reported 76.8 and was refused in forty seconds. That is
        when the threshold stopped being evidence: load in the sixties is the
        NORM for the cheapest CPU flavour in this datacenter, and the pod that
        measured 65 had moved 1.5 GB successfully before it decayed. A gate at
        8.0 refuses every draw, including the ones that work.
        
        So load is recorded and the limit is a backstop for the absurd. What
        actually decides is throughput — measured once before committing, in
        `require_transport_floor`, and continuously during the transfer, in
        `stage_probe`. A proxy for the thing you care about is worth exactly as
        much as its correlation, and this one's was not good enough.
        """
        out = self.run("cat /proc/loadavg; nproc", timeout=120)
        lines = out.stdout.split()
        try:
            load1 = float(lines[0])
            cpus = max(1, int(out.stdout.strip().splitlines()[-1].strip()))
        except (ValueError, IndexError) as exc:
            raise StagingError(
                f"could not read the pod's load or CPU count: {exc}; "
                f"{out.stdout[:200]!r}") from exc
        ratio = load1 / cpus
        health = {"load1": load1, "cpus": cpus, "load_per_cpu": round(ratio, 2),
                  "limit_per_cpu": self.a.max_load_per_cpu}
        say(f"host health: load {load1} over {cpus} cpu(s) = "
            f"{ratio:.1f} per cpu (limit {self.a.max_load_per_cpu})")
        if ratio > self.a.max_load_per_cpu:
            raise StagingError(
                f"the pod's host is oversubscribed: one-minute load {load1} "
                f"across {cpus} vCPU is {ratio:.1f} per vCPU, above the "
                f"{self.a.max_load_per_cpu} limit. A cheap CPU pod lands "
                "wherever the provider has room; draw another rather than "
                "transferring 22 GiB through a thrashing host.")
        self.ev["host_health"] = health
        return health

    def require_transport_floor(self, sample: Path) -> dict:
        """Send a small real slice and refuse a rate that cannot finish.

        The load check above catches the pathology this job actually met. This
        catches the rest: whatever the cause, a rate this low means the job
        cannot complete, and learning that from 32 MiB is better than learning
        it from a per-probe timeout hours later.

        It is a FLOOR, not a measurement. The number it produces is recorded as
        diagnostic and is never promoted into a schedule or a bound — network
        rates move, and re-running this to refine the figure would be spending
        time to manufacture false precision.

        **It measures the WIRE, while the stall detector measures LANDED
        bytes.** This probe sends raw; the transfer sends compressed, and these
        checkpoints compress to about half, so a pod that clears this floor
        lands roughly twice as fast as it reads here. The two figures are
        therefore not comparable and the floor is conservative in the direction
        of refusing — which is the harmless direction, and the gap between a
        healthy draw and a dead one is wide enough that it has never been
        close.
        """
        host, port = self.endpoint                           # type: ignore[misc]
        nbytes = int(self.a.floor_probe_mib) * 1024 * 1024
        cmd = (f"dd if={shlex.quote(str(sample))} bs=1M "
               f"count={self.a.floor_probe_mib} 2>/dev/null | "
               + " ".join(shlex.quote(c) for c in self.ssh_base())
               + " 'cat > /tmp/aad_floor.bin'")
        t0 = time.time()
        res = subprocess.run(["bash", "-c", cmd], capture_output=True,
                             timeout=max(120.0, nbytes / (self.a.floor_mb_per_second * 1e6) * 2))
        dt = max(1e-6, time.time() - t0)
        self.run("rm -f /tmp/aad_floor.bin", timeout=60)
        rate = nbytes / dt / 1e6
        out = {"sampled_mib": self.a.floor_probe_mib,
               "seconds": round(dt, 1),
               "observed_mb_per_second": round(rate, 3),
               "floor_mb_per_second": self.a.floor_mb_per_second,
               "_observed_is_diagnostic": (
                   "one observation of a rate that moves. It is recorded, and "
                   "it is never promoted into a schedule or a bound.")}
        say(f"transport floor: {self.a.floor_probe_mib} MiB in {dt:.0f}s = "
            f"{rate:.3f} MB/s (floor {self.a.floor_mb_per_second})")
        if res.returncode != 0:
            raise StagingError(
                f"the floor probe failed: {(res.stderr or b'')[-300:]!r}")
        if rate < self.a.floor_mb_per_second:
            raise StagingError(
                f"this pod moves {rate:.3f} MB/s, below the "
                f"{self.a.floor_mb_per_second} MB/s floor. At that rate the "
                f"{self.ev['source']['total_bytes'] / 1e9:.1f} GB owed needs "
                f"{self.ev['source']['total_bytes'] / (rate * 1e6) / 3600:.0f} "
                "hours. Draw another pod.")
        self.ev["transport_floor"] = out
        return out

    def require_remote_rsync(self) -> None:
        """The pod's half of the transfer contract. The host's is checked at `$0`.

        A minimal base image is minimal: it is not required to ship rsync, so
        one check and one install attempt, then a refusal that names it.
        """
        if self.run("command -v rsync", timeout=60).returncode == 0:
            return
        say("  rsync absent on the pod; installing")
        self.run("apt-get update -qq && apt-get install -y -qq rsync",
                 timeout=900)
        if self.run("command -v rsync", timeout=60).returncode != 0:
            raise StagingError(
                "rsync is not available on the staging pod and could not be "
                "installed; the transfer has no resumable path")

    # -- teardown ----------------------------------------------------------
    def teardown(self, why: str) -> dict:
        """Terminate and VERIFY. A termination command's exit status is not proof."""
        if not self.pod_id:
            return {"pod_id": None, "terminated": True, "why": why,
                    "_means": "no resource was ever created"}
        say(f"teardown ({why}): terminating {self.pod_id}")
        #: NOTHING IN HERE MAY RAISE. This runs from a `finally`, so an
        #: exception raised while *recording* a teardown destroys the record of
        #: the teardown that already happened — which is how the rehearsal
        #: ended with a terminated pod and no evidence that it had been
        #: terminated. `as_dict` is the provider's own serializer; the previous
        #: spelling invented a `detail` field the type does not have.
        attempts: list[dict] = []
        try:
            attempts = [a.as_dict() for a in self.provider.terminate(self.pod_id)]
        except Exception as exc:                                   # noqa: BLE001
            attempts = [{"method": "terminate", "ok": False,
                         "error": f"{type(exc).__name__}: {exc}"}]
        gone, state = False, None
        deadline = time.time() + 300
        while time.time() < deadline:
            try:
                st = self.provider.get(self.pod_id)
            except Exception as exc:                          # noqa: BLE001
                state = f"unreadable: {type(exc).__name__}: {exc}"
                time.sleep(10)
                continue
            #: `desired_status` and `exists` are what `PodState` actually
            #: carries, and `billing` is the property that reads them the way
            #: every other caller does — unknown counts as still billing. This
            #: read `st.status`, a field the type has never had; the rehearsal
            #: terminated its pod and then could not say so.
            state = st.desired_status if st.exists else "ABSENT"
            if not st.billing:
                gone = True
                break
            time.sleep(10)
        record = {
            "pod_id": self.pod_id, "why": why, "attempts": attempts,
            "verified_gone": gone, "last_status": state,
            "elapsed_minutes": round(self.elapsed_min(), 3),
            "quoted_usd_per_hour": self.price,
            "gpu_usd": round(self.spent(), 4),
            "_verification": (
                "termination is confirmed by polling the control plane until "
                "the pod stops billing; the terminate call's own exit status "
                "has never been treated as proof"),
        }
        say(f"teardown {'CONFIRMED' if gone else 'UNCONFIRMED'} — "
            f"{record['elapsed_minutes']:.1f} min, ${record['gpu_usd']:.4f}")
        return record


# ---------------------------------------------------------------------------
# the transfer
# ---------------------------------------------------------------------------

def remote_hashes(pod: StagingPod, remote_dir: str) -> dict[str, str]:
    """sha256 of every file under `remote_dir`, computed ON the volume.

    `sha256sum` rather than anything Python, so the check does not depend on
    what happens to be installed in the staging image.
    """
    cmd = (f"cd {shlex.quote(remote_dir)} 2>/dev/null && "
           "find . -type f -printf '%P\\n' | LC_ALL=C sort | "
           "tr '\\n' '\\0' | xargs -0 -r sha256sum")
    res = pod.run(cmd, timeout=3600)
    out: dict[str, str] = {}
    for line in res.stdout.splitlines():
        digest, _, name = line.partition("  ")
        if digest and name:
            out[name] = digest
    return out


def remote_bytes(pod: StagingPod, remote_dir: str) -> int:
    """Bytes currently under `remote_dir`. Cheap, and never raises.

    The transfer watcher calls this every few minutes while a pod is billing.
    A control-plane or SSH blip is not evidence that the transfer stopped, so
    an unreadable answer repeats the previous one rather than reading as zero —
    which would look exactly like a stall and kill a healthy transfer.
    """
    res = pod.run(f"du -sb {shlex.quote(remote_dir)} 2>/dev/null | cut -f1",
                  timeout=300)
    text = (res.stdout or "").strip()
    if text.isdigit():
        pod._last_remote_bytes = int(text)
    return getattr(pod, "_last_remote_bytes", 0)


def verify_probe(pod: StagingPod, remote_dir: str, want: dict) -> list[str]:
    """Differences between what the host has and what landed on the volume."""
    got = remote_hashes(pod, remote_dir)
    problems = []
    for rel, meta in want["files"].items():
        if rel not in got:
            problems.append(f"{rel}: absent")
        elif got[rel] != meta["sha256"]:
            problems.append(f"{rel}: {got[rel][:12]}… != {meta['sha256'][:12]}…")
    extra = sorted(set(got) - set(want["files"]))
    if extra:
        problems.append(f"unexpected files on the volume: {extra[:5]}")
    return problems


def stage_probe(pod: StagingPod, name: str, src: Path, remote_dir: str,
                want: dict) -> dict:
    """Move one probe and prove it arrived. Resumes a partial previous copy."""
    pod.check_budget()
    already = verify_probe(pod, remote_dir, want)
    if not already:
        say(f"  {name}: already on the volume and verified — skipped")
        return {"probe_id": name, "action": "skipped", "verified": True,
                "bytes": want["bytes"], "seconds": 0.0}

    host, port = pod.endpoint                                # type: ignore[misc]
    pod.run(f"mkdir -p {shlex.quote(remote_dir)}", timeout=120)
    ssh = ("ssh -p " + port + " -o StrictHostKeyChecking=no "
           "-o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "
           "-o ServerAliveInterval=30 -o ServerAliveCountMax=10")
    #: COMPRESSED IN TRANSIT, and it is not a marginal tuning choice. These
    #: checkpoints are fp32 files holding values produced in bf16, so the low
    #: sixteen mantissa bits of every parameter are zero: a 256 MiB slice
    #: compresses to 52% with rsync's zlib (and to 46% with zstd, which
    #: rsync 3.1 cannot select). The launcher host's uplink is ~0.5-0.65 MB/s
    #: and is a HOST cap, not a per-connection one — three parallel streams
    #: measured ~1.3x, not 3x — so compression is the only lever that matters
    #: and it roughly halves a 13-hour transfer.
    #:
    #: It is safe by construction: every file is re-hashed from the bytes that
    #: landed on the volume and compared to the host's hash, so a compression
    #: path that corrupted anything fails the verification rather than passing
    #: silently.
    cmd = ["rsync", "-a", "--partial", "--inplace", "--compress",
           "-e", ssh, f"{src}/", f"root@{host}:{remote_dir}/"]
    t0 = time.time()
    say(f"  {name}: sending {want['bytes'] / 2**30:.2f} GiB")
    #: WATCHED WHILE IT RUNS, because the failure this met was DECAY rather
    #: than a bad start. The first staging pod passed every startup check,
    #: moved 1.5 GB at 0.3-0.5 MB/s, and then slid to 0.028 MB/s — a rate at
    #: which this job needs ten days. A per-probe timeout does eventually fire
    #: on that, four hours later; a rolling floor fires in ten minutes and
    #: gives the run a chance to draw another pod while the day is still
    #: young.
    #:
    #: The floor is deliberately far below every healthy observation
    #: (0.32-0.91 MB/s) and far above the pathological one. It is a
    #: liveness check, not a performance target, and a transfer that is merely
    #: slow is left alone.
    deadline = t0 + pod.a.probe_timeout_min * 60
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    last_bytes, last_at = remote_bytes(pod, remote_dir), time.time()
    stalled = ""
    while proc.poll() is None:
        time.sleep(min(pod.a.stall_check_seconds, max(1.0, deadline - time.time())))
        if time.time() > deadline:
            stalled = (f"the per-probe deadline of "
                       f"{pod.a.probe_timeout_min:.0f} min elapsed")
            break
        now_bytes, now_at = remote_bytes(pod, remote_dir), time.time()
        window = max(1e-6, now_at - last_at)
        rate = (now_bytes - last_bytes) / window / 1e6
        outstanding = max(0, int(want["bytes"]) - now_bytes)
        say(f"    {name}: {now_bytes / 2**30:.2f} GiB on the volume, "
            f"{rate:.3f} MB/s over the last {window / 60:.0f} min, "
            f"{outstanding / 2**20:.0f} MiB outstanding")
        #: A TRANSFER IN ITS TAIL IS ALIVE, NOT STALLED.
        #:
        #: rsync keeps running after the last byte lands — it verifies the
        #: whole file and settles metadata — and the destination stops growing
        #: while it does. Probes 1 to 4 finished inside a sleep, so no check
        #: ever saw that phase. Probe 5's tail happened to land on a check
        #: boundary: 20 MiB of a 2.22 GiB probe moved in the window, the floor
        #: read 0.046 MB/s, and a transfer that had delivered everything was
        #: killed and its pod torn down and redrawn.
        #:
        #: So the floor only applies while there is real work outstanding. A
        #: transfer genuinely wedged in its tail is not caught here at all —
        #: `--probe-timeout-min` is the backstop for that, and it should be,
        #: because the two failures need different responses.
        if outstanding > pod.a.tail_margin_mib * 2**20 \
                and rate < pod.a.stall_floor_mb_per_second:
            stalled = (f"throughput fell to {rate:.3f} MB/s over "
                       f"{window / 60:.0f} min, below the "
                       f"{pod.a.stall_floor_mb_per_second} MB/s liveness floor")
            break
        last_bytes, last_at = now_bytes, now_at
    if stalled:
        proc.kill()
        proc.wait(timeout=60)
        raise StagingError(
            f"{name}: {stalled}. The partial file is kept and resumes on the "
            "next draw; this pod is not worth continuing through.")
    out_s, err_s = proc.communicate(timeout=120)
    dt = time.time() - t0
    if proc.returncode != 0:
        raise StagingError(
            f"{name}: rsync exited {proc.returncode}: "
            f"{(err_s or out_s)[-600:]}")
    problems = verify_probe(pod, remote_dir, want)
    if problems:
        raise StagingError(f"{name}: the bytes on the volume do not match the "
                           f"host: {problems[:4]}")
    rate = want["bytes"] / dt / 1e6 if dt else 0.0
    say(f"  {name}: verified on the volume — {dt / 60:.1f} min, {rate:.2f} MB/s")
    return {"probe_id": name, "action": "staged", "verified": True,
            "bytes": want["bytes"], "seconds": round(dt, 1),
            "observed_mb_per_second": round(rate, 3)}


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign-root", required=True,
                    help="the campaign's durable root on the launcher host")
    ap.add_argument("--campaign-id", required=True)
    ap.add_argument("--volume-id", required=True)
    ap.add_argument("--volume-gb", type=int, required=True,
                    help="the provisioned size of the network volume; df at "
                         "the mount reports the backing cluster, not the quota")
    ap.add_argument("--data-center", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", required=True, help="evidence record path")
    #: The provider's own small Ubuntu base, not the training image. It carries
    #: an SSH daemon — which is what makes a pod reachable at all — and nothing
    #: else this job needs, so it pulls in seconds. A CPU pod's container disk
    #: is capped per vCPU, and the ~25 GiB CUDA training image does not fit on
    #: the cheapest flavour at all.
    ap.add_argument("--image", default="runpod/base:1.0.2-ubuntu2404")
    ap.add_argument("--disk-gb", type=int, default=15)
    ap.add_argument("--price-cap", type=float, default=0.40,
                    help="refuse a pod quoted above this hourly rate")
    ap.add_argument("--max-usd", type=float, default=3.00,
                    help="CUMULATIVE spend guard across every draw of this run")
    ap.add_argument("--max-draws", type=int, default=8,
                    help="how many pods this run may draw in sequence; one "
                         "create per draw and one active resource at a time")
    ap.add_argument("--min-draw-usd", type=float, default=0.10,
                    help="below this much remaining, a draw is not worth "
                         "starting and the run stops instead")
    ap.add_argument("--startup-limit-min", type=float, default=15.0)
    ap.add_argument("--terminate-after-min", type=float, default=900.0)
    ap.add_argument("--probe-timeout-min", type=float, default=180.0)
    #: HEALTH GATES, one draw each, derived from the failure they prevent.
    #: The first staging pod reported a one-minute load of 65 on a two-vCPU
    #: share and moved 0.028 MB/s. Healthy draws of the same flavour have
    #: measured 0.32-0.49 MB/s.
    ap.add_argument("--max-load-per-cpu", type=float, default=100.0,
                    help="backstop only. Load in the sixties is normal for "
                         "this flavour and a draw at 65 moved bytes fine; "
                         "throughput is what decides, not load")
    ap.add_argument("--floor-mb-per-second", type=float, default=0.15,
                    help="refuse a pod that cannot move bytes this fast; a "
                         "FLOOR, never a schedule or a bound")
    ap.add_argument("--floor-probe-mib", type=int, default=32)
    ap.add_argument("--stall-check-seconds", type=float, default=600.0,
                    help="how often the transfer's own throughput is checked")
    ap.add_argument("--tail-margin-mib", type=float, default=64.0,
                    help="below this much outstanding, the liveness floor does "
                         "not apply: rsync runs on after the last byte lands "
                         "and the destination stops growing while it verifies")
    ap.add_argument("--stall-floor-mb-per-second", type=float, default=0.10,
                    help="a LIVENESS floor, far below every healthy "
                         "observation and far above the pathological one. A "
                         "merely slow transfer is left alone")
    ap.add_argument("--runpod-config",
                    default=str(Path.home() / ".runpod/config.toml"))
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ev: dict = {
        "schema": SCHEMA,
        "authorizes": "nothing",
        "_contract": (
            "One staging run: the campaign's verified probes copied onto a "
            "provider network volume and re-hashed from the bytes that landed "
            "there. This is transport infrastructure. It admits nothing into "
            "the experiment — a continuation still re-identifies every probe "
            "against the identity its campaign announced, on the pod that "
            "consumes it."),
        "run_id": a.run_id,
        "campaign_id": a.campaign_id,
        "volume_id": a.volume_id,
        "data_center": a.data_center,
        "mount": MOUNT,
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "probes": [],
    }

    def save() -> None:
        out.write_text(json.dumps(ev, indent=2, default=str) + "\n")

    campaign_root = Path(a.campaign_root)
    #: FLAT, one directory per probe id, mirroring how `campaign_state` keys
    #: its probes. The producing attempt is recorded per probe in the index
    #: rather than in the path: a continuation restores by probe id, and a
    #: path segment naming an attempt would have to be guessed by whoever
    #: builds the manifest.
    remote_root = f"{MOUNT}/{a.campaign_id}/probes"

    say(f"transport tooling and inventory, before any resource exists")
    ev["local_rsync"] = require_local_rsync()
    inventory = local_inventory(campaign_root)
    ev["source"] = {
        "campaign_root": str(campaign_root),
        "probe_count": len(inventory),
        "total_bytes": sum(v["bytes"] for v in inventory.values()),
        "files_hashed": sum(len(v["files"]) for v in inventory.values()),
        "source_attempts": sorted({v["source_attempt"]
                                   for v in inventory.values()}),
    }
    say(f"{len(inventory)} probes, "
        f"{ev['source']['total_bytes'] / 2**30:.2f} GiB to stage")
    save()

    provider = RunPodProvider(api_key(Path(a.runpod_config)))

    #: ONE CREATE PER DRAW, DRAWS IN SEQUENCE, ONE ACTIVE RESOURCE AT A TIME.
    #:
    #: The cheapest CPU flavour lands wherever the provider has room, and where
    #: it lands decides whether this job takes eight hours or ten days. The
    #: first draw moved 1.5 GB and then decayed to 0.028 MB/s; the second was
    #: refused outright. Neither is a fault to fix — they are draws to abandon
    #: and repeat, and doing that by hand means the transfer only progresses
    #: while someone is watching.
    #:
    #: So the loop is here, bounded three ways: a draw count, a CUMULATIVE
    #: spend cap across every draw, and an immediate stop if any draw cannot
    #: confirm its resource was released. A partial probe survives on the
    #: volume and the next draw resumes it, so redrawing costs only the pod
    #: that was abandoned.
    ev["draws"] = []
    spent = 0.0
    failure = None
    for draw in range(1, a.max_draws + 1):
        remaining = round(a.max_usd - spent, 4)
        if remaining < a.min_draw_usd:
            failure = (f"the ${a.max_usd:.2f} cumulative cap has ${remaining:.4f} "
                       f"left, below the ${a.min_draw_usd:.2f} a draw needs to "
                       "be worth starting")
            say(f"STOPPING: {failure}")
            break

        draw_args = argparse.Namespace(**vars(a))
        draw_args.max_usd = remaining
        pod = StagingPod(draw_args, provider, ev)
        ev["draw"] = draw
        say(f"=== draw {draw} of {a.max_draws}, ${remaining:.4f} of the "
            f"${a.max_usd:.2f} cap remaining ===")
        failure = None
        try:
            pod.create()
            save()
            pod.wait_endpoint()
            pod.wait_ssh()
            pod.require_healthy_host()
            pod.require_remote_rsync()

            #: THE MOUNT, BEFORE ANY DIRECTORY IS CREATED. `mkdir -p /durable/...`
            #: succeeds just as happily on the container's own disk, and `df` would
            #: then report the container disk's free space and look entirely
            #: healthy. The whole run would copy 22 GiB onto a filesystem that dies
            #: with the pod. So the question asked here is "is this a mount point",
            #: which only the attached volume can answer, and it is asked before
            #: anything can create the directory that would mask the answer.
            if pod.run(f"mountpoint -q {MOUNT}", timeout=120).returncode != 0:
                raise StagingError(
                    f"{MOUNT} is not a mount point on the staging pod: the network "
                    "volume did not attach, and anything written there would die "
                    "with the pod")
            #: CAPACITY COMES FROM THE PROVISIONED SIZE, NOT FROM `df`. A network
            #: volume is a quota on a shared cluster filesystem: the rehearsal's
            #: 40 GB volume reported **165732 GiB free**, because `df` was
            #: describing the cluster. A check built on that can never fail and
            #: would have let the real run discover the limit by hitting it — which
            #: is precisely how attempt5 died of ENOSPC.
            need = ev["source"]["total_bytes"]
            if a.volume_gb <= 0:
                raise StagingError(
                    "--volume-gb must state the provisioned size of the network "
                    "volume; df cannot answer this and a guess is not a bound")
            capacity = a.volume_gb * 10**9          # the provider sells GB, not GiB
            used = pod.run(f"du -sb {MOUNT} 2>/dev/null | cut -f1",
                           timeout=1800).stdout.strip()
            occupied = int(used) if used.isdigit() else 0
            ev["volume"] = {"provisioned_gb": a.volume_gb,
                            "capacity_bytes": capacity,
                            "occupied_bytes_at_start": occupied,
                            "need_bytes": need,
                            "_df_is_not_the_quota": (
                                "df at the mount reports the backing cluster, not "
                                "this volume's quota, so free space is derived "
                                "from the provisioned size the provider sold")}
            if occupied + need > capacity:
                raise StagingError(
                    f"the volume is {a.volume_gb} GB and already holds "
                    f"{occupied / 10**9:.2f} GB; {need / 10**9:.2f} GB more would "
                    "exceed it")
            say(f"volume mounted at {MOUNT}: {a.volume_gb} GB provisioned, "
                f"{occupied / 10**9:.2f} GB occupied, {need / 10**9:.2f} GB to write")
            #: AFTER the mount is proven and BEFORE 22 GiB is committed to this
            #: draw. Uses real probe bytes, so it exercises the same read path.
            pod.require_transport_floor(
                Path(next(iter(inventory.values()))["durable_path"])
                / "model.safetensors")
            pod.run(f"mkdir -p {shlex.quote(remote_root)}", timeout=120)

            for name, want in inventory.items():
                pod.check_budget()
                rec = stage_probe(pod, name, Path(want["durable_path"]),
                                  f"{remote_root}/{name}", want)
                rec["source_attempt"] = want["source_attempt"]
                rec["artifact_digest"] = want["artifact_digest"]
                ev["probes"].append(rec)
                save()

            marker = {
                "schema": "aadistill.autoinit.c2_probe_volume_index/v2",
                "campaign_id": a.campaign_id,
                "staged_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "mount": MOUNT,
                "probe_root": remote_root,
                #: PER PROBE, because the launcher must be able to check that the
                #: copy on the volume came from the same attempt its manifest's
                #: identity came from. Probe ids are unique within a campaign and
                #: `campaign_state` collapses them across attempts, so without
                #: this a stale copy from an earlier attempt could sit under the
                #: right name and only fail re-identification on the paid pod.
                "probes": {
                    pid: {"source_attempt": v["source_attempt"],
                          "artifact_digest": v["artifact_digest"],
                          "bytes": v["bytes"], "scored": v["scored"]}
                    for pid, v in sorted(inventory.items())},
                "total_bytes": ev["source"]["total_bytes"],
                "_means": ("these bytes were copied from the campaign's durable "
                           "root and re-hashed here. They are NOT admitted to the "
                           "experiment by this file; the consuming pod does that."),
            }
            pod.run("cat > " + shlex.quote(f"{remote_root}/staged_index.json")
                    + " <<'AADEOF'\n" + json.dumps(marker, indent=2) + "\nAADEOF",
                    timeout=120)
            check = pod.run(f"test -s {shlex.quote(remote_root)}/staged_index.json",
                            timeout=60)
            if check.returncode != 0:
                raise StagingError("the volume index did not write")
            ev["staged_index"] = marker
            ev["staged"] = True
            say("all probes verified on the volume")
        except Exception as exc:                              # noqa: BLE001
            failure = f"{type(exc).__name__}: {exc}"
            ev["error"] = failure
            ev["staged"] = False
            say(f"FAILED: {failure}")
        finally:
            #: The record of a teardown must survive a failure to record it. The
            #: rehearsal terminated its pod and then lost the evidence to an
            #: AttributeError raised while serializing the result, leaving a run
            #: that could not show its own resource was released.
            try:
                ev["teardown"] = pod.teardown("staging complete" if not failure
                                              else "staging failed")
            except Exception as exc:                          # noqa: BLE001
                ev["teardown"] = {
                    "pod_id": pod.pod_id, "verified_gone": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "_means": ("teardown could not be completed or recorded; the "
                               "resource must be reconciled by hand before any "
                               "other resource of this campaign is created")}
                say(f"TEARDOWN RECORDING FAILED: {type(exc).__name__}: {exc}")
            ev["money"] = {
                "gpu_usd": ev["teardown"].get("gpu_usd", 0.0),
                "quoted_usd_per_hour": pod.price,
                "elapsed_minutes": ev["teardown"].get("elapsed_minutes", 0.0),
                "_disk_note": (
                    "container disk and the network volume are billed by the "
                    "provider separately from compute and are reconciled in the "
                    "campaign ledger, not here"),
            }
            ev["finished_utc"] = datetime.now(timezone.utc).isoformat(
                timespec="seconds")
            save()

        record = {"draw": draw, "pod_id": pod.pod_id,
                  "terminal": "STAGED" if ev.get("staged") else "FAILED",
                  "error": ev.get("error"),
                  "teardown": ev.get("teardown"), "money": ev.get("money"),
                  "host_health": ev.pop("host_health", None),
                  "transport_floor": ev.pop("transport_floor", None)}
        ev["draws"].append(record)
        spent = round(spent + float((ev.get("money") or {}).get("gpu_usd", 0.0)), 4)
        ev["cumulative_gpu_usd"] = spent
        save()

        #: AN UNRECONCILED RESOURCE STOPS EVERYTHING. At most one resource of
        #: this job may bill at a time, and a draw that cannot show its pod was
        #: released makes that unknowable — so nothing else is created.
        if not (ev.get("teardown") or {}).get("verified_gone"):
            failure = (f"draw {draw} could not confirm its pod was released; "
                       "no further resource will be created")
            say(f"STOPPING: {failure}")
            break
        if ev.get("staged"):
            break
        say(f"draw {draw} did not finish; ${spent:.4f} spent so far")

    #: A RUN THAT CANNOT SHOW ITS RESOURCE WAS RELEASED HAS NOT SUCCEEDED.
    #: The rehearsal printed "staging COMPLETE" and exited 0 while its own
    #: record said the teardown could not be confirmed — the one outcome an
    #: operator most needs to see, reported as success.
    released = bool((ev.get("teardown") or {}).get("verified_gone"))
    ev["terminal"] = ("ALL_STAGED" if ev.get("staged") and released
                      else "STAGED_BUT_UNRECONCILED" if ev.get("staged")
                      else "FAILED_AND_UNRECONCILED" if not released
                      else "FAILED")
    save()
    print(f"\nstaging {ev['terminal']} — {out}")
    if not released:
        print("RESOURCE STATE UNKNOWN: reconcile the pod before creating "
              "another resource of this campaign.")
        return 12
    return 0 if ev.get("staged") else 11


if __name__ == "__main__":
    raise SystemExit(main())
