#!/usr/bin/env python3
"""Live RunPod control-plane canary. Runs a harmless process, not training.

Verifies on a real disposable pod the ten things the local simulator cannot:
detached start, durable descriptor, log durability off-pod, provider-only
watchdog observation, threshold crossing, the **never-exercised** GraphQL
`podTerminate` fallback, journalled termination, provider-polled disappearance,
manifest + hash verification, and nothing left running.

Two watchdogs, deliberately:

* **safety** — real `runpodctl`, a generous threshold, launched the moment the
  pod exists and detached from this process. It is the honest backstop and is
  not part of the test.
* **test** — launched with `--runpodctl` pointing at a path that does not exist,
  so `_terminate_cli` fails locally and `terminate()` falls through to the
  GraphQL mutation. **No provider state is altered to induce the failure**; only
  this process's view of the CLI changes. Its threshold is set to be crossed
  during the run, so the journal shows a poll below the limit, then
  `hard_limit_reached`, then the attempts, then the verification polls.

If the fallback fails, phase 2 tears the pod down with the real CLI and the
canary reports **FAIL** — a pod is never left running to prove a point.

`--terminate-after` is set as a redundant third layer and does not count toward
a pass.

    PYTHONPATH=src python scripts/pod/canary.py --scr <scratch> --authorized-usd 0.82
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.infrastructure.artifact_gate import (  # noqa: E402
    ArtifactManifest, evaluate_teardown, verify_extracted,
)
from aadistill.infrastructure.log_relay import LogRelay, RelaySpec  # noqa: E402
from aadistill.infrastructure.provider import (  # noqa: E402
    RunPodProvider, read_api_key,
)
from aadistill.infrastructure.remote import (  # noqa: E402
    JobSpec, SSHTarget, probe, start_detached,
)
from aadistill.infrastructure.watchdog import Journal  # noqa: E402

WS = "/workspace"
JOB_DIR = f"{WS}/jobs"
CANARY_DIR = f"{WS}/canary"
EVENTS = f"{CANARY_DIR}/train_log.jsonl"
STATUS = f"{WS}/canary.status"

# The pod-side job: a harmless loop that emits one structured event and one
# marker every few seconds and outlives the watchdog, so "already-emitted events
# survived the pod" is demonstrated rather than asserted.
JOB_TICKS = 50          # x3 s = 150 s: long enough to relay from a live writer,
                        # short enough that the run finishes inside the backstop
JOB_CMD = (
    f"mkdir -p {CANARY_DIR}; i=0; "
    f"while [ $i -lt {JOB_TICKS} ]; do "
    f"printf '{{\"time\":\"%s\",\"event\":\"canary_tick\",\"step\":%d}}\\n' "
    f"\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\" \"$i\" >> {EVENTS}; "
    f"printf '%s MARKER:TICK:%d\\n' \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\" \"$i\" "
    f">> {STATUS}; "
    f"i=$((i+1)); sleep 3; done; "
    # The terminal marker. `final_required` means "the producer said it
    # finished", not merely "the file stopped changing while I looked".
    f"printf '%s MARKER:ALL_DONE\\n' \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\" >> {STATUS}"
)

# Phase A, while the job is still writing: the event stream is a
# `mutable_snapshot`. Its claim is "these bytes are durable", not "this file is
# finished", so growth during archiving is expected and recorded.
SPEC_SNAPSHOT = [
    {"artifact_class": "event_stream", "pattern": "canary/train_log.jsonl",
     "required": True, "min_matches": 1, "min_bytes": 256,
     "lifecycle": "mutable_snapshot"},
    {"artifact_class": "job_descriptor", "pattern": "jobs/*.job.json",
     "required": True, "min_matches": 1, "min_bytes": 32,
     "lifecycle": "final_required"},
]
# Phase B, after the terminal marker: the same stream is now `final_required`
# and must be marker-backed and quiescent. This is the shape a normal E7
# teardown uses.
SPEC_FINAL = [
    {"artifact_class": "event_stream", "pattern": "canary/train_log.jsonl",
     "required": True, "min_matches": 1, "min_bytes": 256,
     "lifecycle": "final_required"},
    {"artifact_class": "job_descriptor", "pattern": "jobs/*.job.json",
     "required": True, "min_matches": 1, "min_bytes": 32,
     "lifecycle": "final_required"},
]
COMPLETION_MARKERS = [{"path": "canary.status", "contains": "MARKER:ALL_DONE"}]

MINI_TREE = [
    ("scripts/pod/collect_artifacts.py", "aad/scripts/pod/collect_artifacts.py"),
    ("src/aadistill/__init__.py", "aad/src/aadistill/__init__.py"),
    ("src/aadistill/infrastructure/__init__.py",
     "aad/src/aadistill/infrastructure/__init__.py"),
    ("src/aadistill/infrastructure/artifact_gate.py",
     "aad/src/aadistill/infrastructure/artifact_gate.py"),
    ("src/aadistill/infrastructure/manifest.py",
     "aad/src/aadistill/infrastructure/manifest.py"),
]


class Canary:
    def __init__(self, args):
        self.a = args
        self.scr = Path(args.scr)
        self.scr.mkdir(parents=True, exist_ok=True)
        self.key = os.environ.get("RUNPOD_API_KEY") or read_api_key(args.runpod_config)
        self.provider = RunPodProvider(self.key)
        self.log_path = self.scr / "canary.log"
        self.ev: dict = {"criteria": {}, "timeline": [], "cost": {}}
        self.pod_id = ""
        self.start_epoch = 0.0
        self.price = None
        self.candidates: list = []
        self.launches: dict[str, int] = {"safety": 0, "test": 0}
        self.backstop_minutes = float(args.backstop_minutes)
        self.safety_minutes = float(args.backstop_minutes) - 5.0
        self.cli = shutil.which("runpodctl") or os.path.expanduser(
            "~/.local/bin/runpodctl")
        if not Path(self.cli).is_file():
            raise SystemExit(
                "runpodctl not found; the canary needs it to create the pod and "
                "as the phase-2 cleanup path")

    # -- helpers -----------------------------------------------------------
    def say(self, msg: str) -> None:
        line = f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")
        self.ev["timeline"].append(
            {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "elapsed_min": round(self.elapsed(), 2), "msg": msg})

    def elapsed(self) -> float:
        return (time.time() - self.start_epoch) / 60 if self.start_epoch else 0.0

    def usd(self) -> float:
        return self.elapsed() / 60 * (self.price or self.a.max_price)

    def gql(self, query: str) -> dict:
        return self.provider._gql(query)

    def record(self, name: str, ok: bool, detail) -> bool:
        self.ev["criteria"][name] = {"pass": bool(ok), "detail": detail}
        self.say(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        return bool(ok)

    # -- 1. price ----------------------------------------------------------
    def price_guard(self) -> bool:
        """Quote every candidate and pick the cheapest inside the cap.

        `securePrice`, never `communityPrice` or
        `lowestPrice.uninterruptablePrice` — `pod create` provisions secure, and
        reading the community floor has already under-reported two runs. The
        canary needs no GPU compute at all, so the cheapest offered card is the
        right one; the list exists because a "Low" stock quote can still fail to
        create, and a wasted create attempt costs time on a metered session.
        """
        quotes = []
        for gpu in [g.strip() for g in self.a.gpu.split(",") if g.strip()]:
            d = self.gql('query { gpuTypes(input:{id:"%s"}) { id securePrice '
                         'lowestPrice(input:{gpuCount:1}) { stockStatus } } }' % gpu)
            rows = (d.get("data") or {}).get("gpuTypes") or []
            if not rows:
                self.say(f"  {gpu}: not offered"); continue
            sp = rows[0].get("securePrice")
            st = (rows[0].get("lowestPrice") or {}).get("stockStatus")
            self.say(f"  {gpu}: securePrice ${sp}/h, stock {st}")
            if sp is not None and sp <= self.a.max_price:
                quotes.append((sp, gpu, st))
        quotes.sort()
        self.ev["quotes"] = [{"gpu": g, "secure_price_per_hour": s,
                              "stock_status": st} for s, g, st in quotes]
        if not quotes:
            self.say(f"ABORT: nothing offered at or under "
                     f"${self.a.max_price}/h"); return False
        self.candidates = quotes
        self.price = quotes[0][0]
        self.ev["quoted_price_per_hour"] = self.price
        self.ev["stock_status"] = quotes[0][2]
        self.say(f"selected {quotes[0][1]} at ${self.price}/h")
        # The backstop is DERIVED from the quote, never the other way round.
        # The authorization is a dollar figure; minutes are whatever fits inside
        # it. If that leaves too little time to complete the canary, stop and
        # report — do not widen the backstop to make the run fit.
        fitted = self.a.authorized_usd / self.price * 60
        self.ev["backstop_minutes_fitted"] = round(fitted, 2)
        if fitted < self.a.min_backstop_minutes:
            self.say(
                f"ABORT: ${self.a.authorized_usd} buys only {fitted:.1f} min at "
                f"${self.price}/h, under the {self.a.min_backstop_minutes:.0f} "
                "min this canary needs. Reporting rather than changing the "
                "backstop.")
            return False
        self.backstop_minutes = min(self.a.backstop_minutes, fitted)
        self.safety_minutes = max(1.0, self.backstop_minutes - 5.0)
        worst = self.backstop_minutes / 60 * self.price
        self.ev["backstop_minutes"] = round(self.backstop_minutes, 2)
        self.ev["worst_case_at_backstop_usd"] = round(worst, 4)
        self.say(f"backstop {self.backstop_minutes:.1f} min = ${worst:.3f} "
                 f"(authorized ${self.a.authorized_usd}); safety watchdog at "
                 f"{self.safety_minutes:.1f} min")
        return True

    # -- 2. create ---------------------------------------------------------
    def create(self) -> bool:
        deadline = (datetime.now(timezone.utc)
                    + timedelta(minutes=self.backstop_minutes))
        pid, chosen, price = "", "", None
        for sp, gpu, _stock in self.candidates:
            raw = subprocess.run(
                [self.cli, "pod", "create", "--image", self.a.image,
                 "--gpu-id", gpu, "--gpu-count", "1",
                 "--container-disk-in-gb", "20", "--volume-in-gb", "0",
                 "--ports", "22/tcp", "--name", "aadistill-canary",
                 "--terminate-after", deadline.strftime("%Y-%m-%dT%H:%M:%SZ")],
                capture_output=True, text=True, timeout=300)
            (self.scr / f"create_raw_{gpu.replace(' ', '_')}.txt").write_text(
                raw.stdout + raw.stderr)
            try:
                pid = json.loads(raw.stdout).get("id", "")
            except Exception:
                m = re.search(r'"id"\s*:\s*"([^"]+)"', raw.stdout + raw.stderr)
                pid = m.group(1) if m else ""
            if pid:
                chosen, price = gpu, sp
                # The rate the pod ACTUALLY provisioned at, re-checked rather
                # than assumed from the quote.
                try:
                    actual = json.loads(raw.stdout).get("costPerHr")
                    if actual is not None:
                        price = float(actual)
                except Exception:
                    pass
                break
            self.say(f"  create on {gpu} failed: "
                     f"{(raw.stdout + raw.stderr).strip()[:200]}")
        if not pid:
            self.say("ABORT: every candidate GPU failed to create")
            return False
        self.price = price
        self.ev["gpu"] = chosen
        self.ev["actual_price_per_hour"] = price
        if price > self.a.max_price:
            self.say(f"ABORT: provisioned at ${price}/h, above "
                     f"${self.a.max_price}/h — deleting")
            subprocess.run([self.cli, "remove", "pod", pid],
                           capture_output=True, timeout=120)
            return False
        self.start_epoch = time.time()
        self.pod_id = pid
        (self.scr / "pod_id").write_text(pid)
        (self.scr / "pod_start_epoch").write_text(str(self.start_epoch))
        self.ev["pod_id"] = pid
        self.ev["terminate_after_utc"] = deadline.strftime("%Y-%m-%dT%H:%M:%SZ")
        self.say(f"created {pid}; --terminate-after {self.ev['terminate_after_utc']} "
                 "(redundant third layer, does not count toward PASS)")
        return True

    # -- 3. watchdogs ------------------------------------------------------
    def launch_watchdog(self, tag: str, hard_minutes: float,
                        runpodctl: str | None) -> Path:
        journal = self.scr / f"watchdog_{tag}.jsonl"
        cmd = [sys.executable, str(REPO_ROOT / "scripts/pod/watchdog.py"),
               "--pod-id", self.pod_id,
               "--session-start-epoch", str(self.start_epoch),
               "--price-per-hour", str(self.price),
               "--hard-minutes", str(hard_minutes),
               "--authorized-usd", str(self.a.authorized_usd),
               "--journal", str(journal),
               "--poll-seconds", "20", "--verify-delay-seconds", "10",
               "--terminate-rounds", "3", "--verify-polls", "3"]
        if runpodctl:
            cmd += ["--runpodctl", runpodctl]
        env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
        out = open(self.scr / f"watchdog_{tag}.out", "w")
        subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, cwd=REPO_ROOT, env=env,
                         start_new_session=True)
        self.launches[tag] = self.launches.get(tag, 0) + 1
        self.say(f"watchdog[{tag}] detached (launch #{self.launches[tag]}) — "
                 f"hard {hard_minutes:.1f} min, runpodctl={runpodctl or 'real'}")
        return journal

    # -- 4. endpoint -------------------------------------------------------
    def wait_endpoint(self) -> tuple[str, str] | None:
        deadline = time.time() + self.a.startup_limit_min * 60
        i = 0
        while time.time() < deadline:
            d = self.gql('query { pod(input:{podId:"%s"}) { runtime { ports '
                         '{ ip publicPort privatePort type } } } }' % self.pod_id)
            rt = ((d.get("data") or {}).get("pod") or {}).get("runtime")
            if rt:
                for p in rt.get("ports") or []:
                    if p.get("privatePort") == 22 and p.get("type") == "tcp":
                        self.say(f"TCP 22 at {p['ip']}:{p['publicPort']} "
                                 f"after {self.elapsed():.1f} min")
                        return str(p["ip"]), str(p["publicPort"])
            i += 1
            if i % 6 == 0:
                self.say(f"  starting ({i * 10}s) — ${self.usd():.2f}")
            time.sleep(10)
        return None

    def collect(self, target, host, scp, phase: str, spec_name: str, *,
                settle: float, markers: str) -> dict:
        """One full collection round, pod-side then dev-box-side.

        Returns data rather than asserting, so the caller decides what each
        phase was supposed to prove: the snapshot round claims durability, the
        final round claims completeness, and they are not the same claim.
        """
        cc = ("cd /workspace/aad && PYTHONPATH=/workspace/aad/src python3 "
              "scripts/pod/collect_artifacts.py")
        mpath = f"{WS}/manifest_{phase}.json"
        apath = f"{WS}/artifacts_{phase}.tar.gz"
        cmd = (f"{cc} manifest --root {WS} --spec {WS}/{spec_name} "
               f"--out {mpath} --settle-seconds {settle}")
        if markers:
            cmd += f" --completion-markers {markers}"
        man = target.run(cmd, timeout=240)
        self.say(f"  [{phase}] manifest rc={man.returncode} :: "
                 f"{man.stdout.strip().splitlines()[-1] if man.stdout.strip() else ''}")
        arc = target.run(f"{cc} archive --manifest {mpath} --out {apath}",
                         timeout=240)
        ver = target.run(f"{cc} verify-archive --manifest {mpath} "
                         f"--archive {apath}", timeout=240)
        store = self.scr / f"store_{phase}"
        store.mkdir(exist_ok=True)
        for remote, local in ((mpath, store / "manifest.json"),
                              (apath, store / "artifacts.tar.gz")):
            subprocess.run(scp + [f"root@{host}:{remote}", str(local)],
                           capture_output=True, timeout=300)
        out = {"phase": phase, "manifest_rc": man.returncode,
               "archive_rc": arc.returncode, "verify_archive_rc": ver.returncode,
               "pod_manifest_stdout": man.stdout.strip()[-600:],
               "local_ok": False, "manifest": {}, "gate": None}
        if not (store / "manifest.json").is_file():
            return out
        import tarfile
        extract = store / "extracted"
        extract.mkdir(exist_ok=True)
        try:
            with tarfile.open(store / "artifacts.tar.gz") as tar:
                tar.extractall(extract, filter="data")
            manifest = ArtifactManifest.load(store / "manifest.json")
            problems = verify_extracted(extract, manifest)
            out["local_ok"] = not problems
            out["local_problems"] = problems
            out["manifest"] = {
                "ok": manifest.ok,
                "final_streams_quiescent": manifest.final_streams_quiescent,
                "final_entries": len(manifest.final_entries()),
                "snapshot_entries": len(manifest.snapshot_entries()),
                "appended_during_archive": manifest.appended_during_archive,
                "still_being_written": manifest.still_being_written,
                "completion_marker_failures": manifest.completion_marker_failures,
                "entries": [{"path": e.path, "lifecycle": e.lifecycle,
                             "size_bytes": e.size_bytes,
                             "sha256": e.sha256[:16] + "…"}
                            for e in manifest.entries],
            }
            out["gate"] = evaluate_teardown({
                "training_complete": True, "evaluation_complete": True,
                "artifact_manifest_created": True,
                "required_files_present": manifest.ok,
                "final_streams_quiescent": manifest.final_streams_quiescent,
                "archive_created": True,
                "archive_contents_verified": ver.returncode == 0,
                "transfer_complete": True,
                "local_hashes_verified": out["local_ok"],
                "checkpoint_hashes_matched": True,
                "report_inputs_verified": True}).as_dict()
        except Exception as exc:  # noqa: BLE001
            out["error"] = f"{type(exc).__name__}: {exc}"
        return out

    # -- 5-12 --------------------------------------------------------------
    def run(self) -> bool:
        if not self.price_guard() or not self.create():
            return False
        safety = self.launch_watchdog("safety", self.safety_minutes, None)
        self.ev["watchdog_safety_journal"] = str(safety)

        ep = self.wait_endpoint()
        if not ep:
            self.say("ABORT: no TCP 22 in time; the safety watchdog owns cleanup")
            return False
        host, port = ep
        target = SSHTarget(host, port)
        for _ in range(30):
            if target.run("true", timeout=30).returncode == 0:
                break
            time.sleep(10)
        self.say(f"ssh reachable — ${self.usd():.2f}")

        scp = ["scp", "-P", port, "-o", "StrictHostKeyChecking=no",
               "-o", "UserKnownHostsFile=/dev/null"]
        target.run(f"mkdir -p {JOB_DIR} {CANARY_DIR} "
                   f"/workspace/aad/scripts/pod "
                   f"/workspace/aad/src/aadistill/infrastructure", timeout=60)
        for local, remote in MINI_TREE:
            subprocess.run(scp + [str(REPO_ROOT / local),
                                  f"root@{host}:{WS}/{remote}"],
                           capture_output=True, timeout=120)
        for name, payload in (("spec_snapshot.json", SPEC_SNAPSHOT),
                              ("spec_final.json", SPEC_FINAL),
                              ("markers.json", COMPLETION_MARKERS)):
            local = self.scr / name
            local.write_text(json.dumps(payload, indent=2))
            subprocess.run(scp + [str(local), f"root@{host}:{WS}/{name}"],
                           capture_output=True, timeout=120)

        # -- criterion 1-2: detached start + durable descriptor
        job_spec = JobSpec(job_id="canary", workdir=WS, command=JOB_CMD,
                           job_dir=JOB_DIR, log_path=f"{WS}/canary_run.log",
                           status_path=STATUS)
        t0 = time.monotonic()
        job = start_detached(target, job_spec, start_timeout=120,
                             verify_timeout=60)
        start_secs = time.monotonic() - t0
        self.record("detached_launch_returns_promptly", start_secs < 120,
                    {"seconds": round(start_secs, 2), "pid": job.pid,
                     "start_channel_closed": job.start_channel_closed,
                     "confirmed_by": job.confirmed_by})
        desc = target.run(f"cat {job.descriptor_path}", timeout=60)
        alive = probe(target, job)[0]
        self.record("durable_job_descriptor",
                    desc.returncode == 0 and '"pid"' in desc.stdout
                    and alive == "ALIVE",
                    {"descriptor": desc.stdout.strip()[:300], "liveness": alive})

        # -- criterion 3: structured logs off the pod while the writer is live
        relay_spec = RelaySpec(EVENTS, "canary.train_log.jsonl")
        relay = LogRelay(target, (relay_spec,), self.scr / "relay")
        cycles = []
        for k in range(4):
            time.sleep(15)
            r = relay.sync_once()
            cycles.append({"cycle": k + 1, **r.as_dict()})
            self.say(f"  relay cycle {k + 1}: {r.as_dict()}")
        live = probe(target, job)[0]
        events = relay.recovered_events(relay_spec)
        self.ev["relay_cycles"] = cycles
        self.record("structured_logs_relayed_while_process_active",
                    len(events) >= 3 and live == "ALIVE"
                    and all(e.get("event") == "canary_tick" for e in events),
                    {"events_synced": len(events), "job_liveness": live,
                     "first": events[0] if events else None,
                     "last": events[-1] if events else None})

        # -- criterion 4: the provider-only watchdog sees a live billing pod
        rows = Journal(safety).records()
        polls = [r for r in rows if r["event"] == "poll"]
        self.record("watchdog_starts_and_polls_live_billing_pod",
                    any(p.get("pod_billing") and p.get("desired_status") == "RUNNING"
                        for p in polls),
                    {"polls": len(polls), "last": polls[-1] if polls else None})

        # -- criterion 9: mutable-snapshot semantics, writer still active
        snap = self.collect(target, host, scp, "snapshot", "spec_snapshot.json",
                            settle=0.0, markers="")
        snap_ok = (snap["manifest_rc"] == 0 and snap["archive_rc"] == 0
                   and snap["verify_archive_rc"] == 0 and snap["local_ok"]
                   and snap["manifest"].get("snapshot_entries", 0) >= 1)
        self.record("mutable_snapshot_semantics_without_hash_races", snap_ok,
                    snap)

        # -- criterion 10: final_required semantics, after the terminal marker
        self.say("waiting for the job's terminal marker")
        done = False
        deadline = time.time() + self.a.job_wait_min * 60
        while time.time() < deadline:
            r = target.run(f"grep -c ALL_DONE {STATUS} || true", timeout=60)
            if r.stdout.strip() not in ("", "0"):
                done = True
                break
            time.sleep(10)
        self.say(f"  terminal marker {'seen' if done else 'NOT seen'} at "
                 f"{self.elapsed():.1f} min")
        final = self.collect(target, host, scp, "final", "spec_final.json",
                             settle=self.a.settle_seconds,
                             markers=f"{WS}/markers.json")
        gate = final.get("gate")
        final_ok = (done and final["manifest_rc"] == 0
                    and final["archive_rc"] == 0
                    and final["verify_archive_rc"] == 0 and final["local_ok"]
                    and final["manifest"].get("final_streams_quiescent") is True
                    and bool(gate and gate["allowed"]) and not gate["emergency"])
        self.record("final_artifact_and_hash_verification_under_declared_semantics",
                    final_ok, final)

        # -- criteria 5-8: the test watchdog, forced onto the GraphQL fallback
        threshold = self.elapsed() + self.a.cross_after_min
        self.say(f"launching the test watchdog: threshold {threshold:.2f} min "
                 f"(now {self.elapsed():.2f}), CLI path forced to fail")
        test_j = self.launch_watchdog("test", threshold, self.a.broken_runpodctl)
        self.ev["watchdog_test_journal"] = str(test_j)
        deadline = time.time() + self.a.watchdog_wait_min * 60
        reason = None
        while time.time() < deadline:
            time.sleep(10)
            rows = Journal(test_j).records()
            if any(r["event"] == "terminated" for r in rows):
                reason = "terminated"
                break
            if any(r["event"] == "TERMINATION_FAILED" for r in rows):
                reason = "termination_failed"
                break
        rows = Journal(test_j).records()
        (self.scr / "watchdog_test_records.json").write_text(
            json.dumps(rows, indent=2))
        self.ev["watchdog_test_reason"] = reason
        self.ev["watchdog_launches"] = dict(self.launches)

        attempts = [r for r in rows if r["event"] == "terminate_attempt"]
        cli = [a for r in attempts for a in r["attempts"]
               if a["method"] == "runpodctl remove pod"]
        gql = [a for r in attempts for a in r["attempts"]
               if a["method"] == "graphql podTerminate"]
        crossed = [r for r in rows if r["event"] == "hard_limit_reached"]
        under = [r for r in rows if r["event"] == "poll"
                 and r.get("over_hard_limit") is False]

        # 5 — the deliberately broken primary path fails as intended
        self.record("broken_primary_runpodctl_fails_as_intended",
                    bool(cli) and not any(a["ok"] for a in cli)
                    and all(a["verified_transport"] for a in cli),
                    {"attempts": cli,
                     "path": self.a.broken_runpodctl,
                     "note": "no provider state altered; only this process's "
                             "view of the CLI"})
        # 6 — the same watchdog invokes the GraphQL fallback automatically
        self.record("graphql_fallback_invoked_automatically",
                    bool(gql) and any(a["ok"] for a in gql) and bool(crossed)
                    and bool(under) and self.launches["test"] == 1,
                    {"attempts": gql,
                     "test_watchdog_launches": self.launches["test"],
                     "polls_below_limit": len(under),
                     "hard_limit_reached": crossed[0] if crossed else None})
        # 7 — provider polling verifies the pod is gone
        verifies = [r for r in rows if r["event"] == "terminate_verify"]
        gone_rows = [r for r in verifies if r.get("pod_exists") is False
                     and r.get("billing") is False]
        self.record("provider_polling_verifies_disappearance",
                    bool(gone_rows)
                    and gone_rows[0].get("desired_status") == "TERMINATED",
                    {"verify_polls": len(verifies),
                     "first_gone": gone_rows[0] if gone_rows else None})
        # 8 — the journal durably records the whole sequence
        seq = [r["event"] for r in rows]
        required_seq = ["watchdog_start", "poll", "hard_limit_reached",
                        "terminate_attempt", "terminate_verify", "terminated"]
        self.record("journal_records_complete_termination_sequence",
                    all(e in seq for e in required_seq)
                    and Path(test_j).is_file(),
                    {"events": seq, "journal": str(test_j),
                     "bytes": Path(test_j).stat().st_size
                     if Path(test_j).is_file() else 0})

        # -- phase 2: never leave a pod running to prove a point
        final_state = self.provider.get(self.pod_id)
        if final_state.billing:
            self.say("test watchdog did not remove the pod — PHASE 2: real CLI")
            self.ev["phase_2_invoked"] = True
            subprocess.run([self.cli, "remove", "pod", self.pod_id],
                           capture_output=True, timeout=120)
            for _ in range(12):
                time.sleep(10)
                final_state = self.provider.get(self.pod_id)
                if not final_state.billing:
                    break
        self.ev["final_pod_state"] = {
            "exists": final_state.exists,
            "desired_status": final_state.desired_status,
            "billing": final_state.billing, "error": final_state.error}

        # 11 — the orchestration got here on its own
        self.record(
            "orchestration_completed_without_human_repair",
            all(c["pass"] for c in self.ev["criteria"].values())
            and self.launches == {"safety": 1, "test": 1}
            and not self.ev.get("phase_2_invoked"),
            {"watchdog_launches": dict(self.launches),
             "phase_2_invoked": bool(self.ev.get("phase_2_invoked")),
             "note": "one launch command; no manual watchdog restart, ssh "
                     "repair or substitution"})
        # 12 — nothing left running
        self.record("no_pod_remains_running", not final_state.billing,
                    self.ev["final_pod_state"])

        self.ev["cost"] = {
            "price_per_hour": self.price,
            "backstop_minutes": round(self.backstop_minutes, 2),
            "elapsed_minutes": round(self.elapsed(), 2),
            "actual_usd": round(self.usd(), 4),
            "authorized_usd": self.a.authorized_usd,
            "within_backstop": self.usd() <= self.a.authorized_usd,
        }
        return all(c["pass"] for c in self.ev["criteria"].values())


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scr", required=True)
    ap.add_argument("--gpu",
                    default="NVIDIA RTX A4000,NVIDIA RTX A4500,"
                            "NVIDIA RTX 2000 Ada Generation,NVIDIA RTX A5000,"
                            "NVIDIA RTX A6000,NVIDIA GeForce RTX 4090",
                    help="comma-separated preference list, cheapest first; the "
                         "canary needs no GPU compute")
    ap.add_argument("--image",
                    default="runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404")
    ap.add_argument("--max-price", type=float, default=0.40,
                    help="above this, $0.12 does not buy enough minutes")
    ap.add_argument("--authorized-usd", type=float, default=0.12)
    ap.add_argument("--backstop-minutes", type=float, default=30.0,
                    help="upper bound; the effective backstop is the smaller of "
                         "this and what --authorized-usd buys at the live quote")
    ap.add_argument("--min-backstop-minutes", type=float, default=20.0,
                    help="below this the canary cannot complete; abort and "
                         "report rather than widening the backstop")
    ap.add_argument("--job-wait-min", type=float, default=6.0)
    ap.add_argument("--settle-seconds", type=float, default=6.0)
    ap.add_argument("--cross-after-min", type=float, default=1.5,
                    help="how far above 'now' to set the test watchdog's "
                         "threshold, so it is genuinely crossed during the run")
    ap.add_argument("--watchdog-wait-min", type=float, default=12.0)
    ap.add_argument("--startup-limit-min", type=float, default=15.0)
    ap.add_argument("--broken-runpodctl",
                    default="/nonexistent/runpodctl-canary-forced-failure")
    ap.add_argument("--runpod-config",
                    default=os.path.expanduser("~/.runpod/config.toml"))
    ap.add_argument("--out", default="logs/e7_canary_rerun_evidence.json")
    args = ap.parse_args()

    c = Canary(args)
    ok = False
    try:
        ok = c.run()
    except Exception as exc:  # noqa: BLE001 — must still tear down and report
        c.ev["driver_error"] = f"{type(exc).__name__}: {exc}"
        c.say(f"DRIVER ERROR: {type(exc).__name__}: {exc}")
        if c.pod_id:
            subprocess.run([c.cli, "remove", "pod", c.pod_id],
                           capture_output=True, timeout=120)
            time.sleep(10)
            st = c.provider.get(c.pod_id)
            c.ev["final_pod_state"] = {"exists": st.exists,
                                       "desired_status": st.desired_status,
                                       "billing": st.billing}
    c.ev["passed"] = bool(ok)
    c.ev["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(c.ev, indent=2) + "\n")
    print(f"\ncanary {'PASSED' if ok else 'FAILED'} — evidence at {out}")
    return 0 if ok else 10


if __name__ == "__main__":
    raise SystemExit(main())
