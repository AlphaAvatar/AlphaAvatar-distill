#!/usr/bin/env python3
"""List a network volume before it is deleted, and bring home what only it holds.

    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/stages/stage-0/calibration_volume/volume_audit.py \
        --volume-id <id> --data-center <dc> --run-id volaudit_<utc> \
        --store <out-of-tree dir> --record-dir logs/.../runs/volaudit_<utc>

The study's volumes are retired under AGENTS.md 2.5 once the extension is
analysed. Retirement deletes WEIGHTS; it never deletes evidence. The sessions'
collectors fetched the records they knew about, but not every small file a
session wrote to the volume (d2's own retired_states.json, its tombstone for 88
states, was written there in d4's session and fetched nowhere). So before a
volume goes, one CPU pod (the cheapest the provider sells) attaches it and:

* lists every file with its size -- the tombstone's member list;
* hashes every file of at most --small-max-bytes, and every larger file whose
  path matches --hash-re (the recovered students, whose sha256 the unit records
  claim), on the pod;
* streams every small file home into --store and re-hashes it here.

Weights stay where they are; nothing is uploaded. The pod comes from the
staging tool's StagingPod: one create call, a hard dollar guard on the quoted
price, a provider-side terminate-after, and a teardown verified by polling.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

from aadistill.infrastructure.provider import RunPodProvider  # noqa: E402
from shared.pod.stage_c2_probes_to_volume import (  # noqa: E402
    MOUNT, StagingPod, api_key, say, sha256_file)

SCHEMA = "aadistill.stage0.calibration_volume.volume_audit/v1"


class AuditPod(StagingPod):
    """StagingPod with an optional GPU: US-TX-3 sells no CPU pod (2026-10-21), and
    a volume can only be attached in its own datacenter."""

    def create(self) -> None:
        if not self.a.gpu_id:
            return super().create()
        deadline = datetime.now(timezone.utc) + timedelta(minutes=self.a.terminate_after_min)
        cmd = ["runpodctl", "pod", "create", "--gpu-id", self.a.gpu_id, "--gpu-count", "1",
               "--image", self.a.image, "--container-disk-in-gb", str(self.a.disk_gb),
               "--network-volume-id", self.a.volume_id, "--volume-mount-path", MOUNT,
               "--data-center-ids", self.a.data_center, "--ports", "22/tcp",
               "--name", f"aad-{self.a.run_id}",
               "--terminate-after", deadline.strftime("%Y-%m-%dT%H:%M:%SZ")]
        say("create: " + " ".join(shlex.quote(c) for c in cmd))
        raw = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        self.ev["create_raw"] = (raw.stdout + raw.stderr)[-4000:]
        try:
            doc = json.loads(raw.stdout)
        except ValueError:
            doc = {}
        pid = doc.get("id") or ""
        m = None if pid else re.search(r'"id"\s*:\s*"([^"]+)"', raw.stdout + raw.stderr)
        pid = pid or (m.group(1) if m else "")
        if not pid:
            raise RuntimeError(f"no pod id from the provider: {(raw.stdout + raw.stderr)[:600]}")
        self.pod_id, self.start_epoch = pid, time.time()
        self.ev["pod_id"] = pid
        self.ev["pod_started_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        quoted = doc.get("costPerHr")
        self.price = float(quoted) if quoted is not None else self.a.price_cap
        self.ev["quoted_usd_per_hour"] = self.price
        say(f"pod {pid} ({self.a.gpu_id}) created at ${self.price:.4f}/h")
        if self.price > self.a.price_cap:
            raise RuntimeError(f"provider quoted ${self.price:.4f}/h above the "
                               f"${self.a.price_cap:.4f}/h cap")


def put_list(pod: StagingPod, remote: str, paths: list[str]) -> None:
    subprocess.run(pod.ssh_base() + [f"cat > {shlex.quote(remote)}"],
                   input="\n".join(paths) + "\n", text=True, check=True, timeout=120)


def remote_sha256(pod: StagingPod, paths: list[str]) -> dict[str, str]:
    if not paths:
        return {}
    put_list(pod, "/tmp/hash.list", paths)
    r = pod.run(f"cd {MOUNT} && xargs -d '\\n' -a /tmp/hash.list sha256sum", timeout=7200)
    if r.returncode != 0:
        raise RuntimeError(f"sha256sum on the pod failed: {r.stderr[-800:]}")
    out = {}
    for line in r.stdout.splitlines():
        digest, path = line.split("  ", 1)
        out[path] = digest
    return out


def audit(pod: StagingPod, a, ev: dict) -> None:
    r = pod.run(f"cd {MOUNT} && find . -type f -printf '%s\\t%p\\n'", timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(f"listing failed: {r.stderr[-800:]}")
    files = []
    for line in r.stdout.splitlines():
        size, path = line.split("\t", 1)
        files.append({"path": path.removeprefix("./"), "bytes": int(size)})
    files.sort(key=lambda f: f["path"])
    say(f"listed {len(files)} files, {sum(f['bytes'] for f in files) / 1e9:.2f} GB")
    small = [f["path"] for f in files if f["bytes"] <= a.small_max_bytes]
    named = [f["path"] for f in files
             if f["bytes"] > a.small_max_bytes and re.search(a.hash_re, f["path"])]
    digests = remote_sha256(pod, small + named)
    say(f"hashed {len(digests)} files on the pod — ${pod.spent():.4f}")

    # small files come home through one tar stream, then are re-hashed here
    tar_path = a.store / "small_files.tar"
    put_list(pod, "/tmp/small.list", small)
    with tar_path.open("wb") as fh:
        p = subprocess.run(pod.ssh_base() + [f"tar -C {MOUNT} -cf - -T /tmp/small.list"],
                           stdout=fh, stderr=subprocess.PIPE, timeout=7200)
    if p.returncode != 0:
        raise RuntimeError(f"tar stream failed: {p.stderr[-800:]!r}")
    dest = a.store / "files"
    with tarfile.open(tar_path) as tf:
        tf.extractall(dest, filter="data")
    tar_path.unlink()
    bad = [s for s in small if sha256_file(dest / s) != digests[s]]
    if bad:
        raise RuntimeError(f"{len(bad)} fetched files do not match the pod's sha256: {bad[:5]}")
    say(f"fetched {len(small)} small files, all re-hashed equal")
    for f in files:
        f["sha256"] = digests.get(f["path"])
        f["fetched"] = f["path"] in set(small)
    ev["files"] = files
    ev["totals"] = {"files": len(files), "bytes": sum(f["bytes"] for f in files),
                    "fetched_files": len(small),
                    "fetched_bytes": sum(f["bytes"] for f in files if f["fetched"]),
                    "hashed_files": len(digests)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--volume-id", required=True)
    ap.add_argument("--data-center", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--store", type=Path, required=True, help="out-of-tree: fetched files land here")
    ap.add_argument("--record-dir", type=Path, required=True, help="the run's directory under logs/")
    ap.add_argument("--small-max-bytes", type=int, default=256 * 2**20)
    ap.add_argument("--hash-re", default=r"(^|/)model\.safetensors$")
    ap.add_argument("--gpu-id", default=None, help="a GPU pod where the datacenter sells no CPU pod")
    ap.add_argument("--image", default="runpod/base:1.0.2-ubuntu2404")
    ap.add_argument("--disk-gb", type=int, default=20)
    ap.add_argument("--price-cap", type=float, default=0.40)
    ap.add_argument("--max-usd", type=float, default=0.75)
    ap.add_argument("--startup-limit-min", type=float, default=15.0)
    ap.add_argument("--terminate-after-min", type=float, default=90.0)
    ap.add_argument("--runpod-config", type=Path, default=Path.home() / ".runpod/config.toml")
    a = ap.parse_args(argv)
    a.store.mkdir(parents=True, exist_ok=True)
    pod_args = SimpleNamespace(gpu_id=a.gpu_id, max_usd=a.max_usd, terminate_after_min=a.terminate_after_min,
                               image=a.image, disk_gb=a.disk_gb, volume_id=a.volume_id,
                               data_center=a.data_center, run_id=a.run_id,
                               price_cap=a.price_cap, startup_limit_min=a.startup_limit_min)
    ev = {"schema": SCHEMA, "run_id": a.run_id, "volume_id": a.volume_id,
          "data_center_id": a.data_center, "mount": MOUNT, "store": str(a.store),
          "gpu_id": a.gpu_id, "small_max_bytes": a.small_max_bytes, "hash_re": a.hash_re,
          "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    pod = AuditPod(pod_args, RunPodProvider(api_key(a.runpod_config)), ev)
    status, err = "FAILED", None
    try:
        pod.create()
        pod.wait_endpoint()
        pod.wait_ssh()
        audit(pod, a, ev)
        status = "AUDITED"
    except Exception as exc:                                       # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
        say(f"audit failed: {err}")
    finally:
        ev["teardown"] = pod.teardown(status)
    ev["status"], ev["error"] = status, err
    ev["ended_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (a.store / "volume_audit.json").write_text(json.dumps(ev, indent=1) + "\n")
    rec = a.record_dir / "evidence"
    rec.mkdir(parents=True, exist_ok=True)
    (rec / "volume_audit.json").write_text(json.dumps(ev, indent=1) + "\n")
    say(f"{status}: {rec / 'volume_audit.json'}")
    return 0 if status == "AUDITED" and ev["teardown"].get("verified_gone") else 1


if __name__ == "__main__":
    raise SystemExit(main())
