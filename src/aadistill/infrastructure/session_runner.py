"""One runner for every paid pod session. It consumes a spec and nothing else.

This is `scripts/pod/autoinit_preflight_launch.py`'s `Preflight` **transformed**,
not rewritten. The flow that has been verified on hardware — detached start with
a durable descriptor, an independent provider-side watchdog, continuous log
relay, four budget thresholds, the artifact gate, provider-confirmed teardown —
survives here in substance, step for step. What changed is where the *session's*
facts come from: they arrive in a :class:`~aadistill.infrastructure.session.SessionSpec`
instead of in a subclass's overrides plus a mutated module global.

The runner is never subclassed and never mutates a module global. Those two
sentences are the whole point: three paid pods were lost to a session inheriting
a requirement it never declared, and a contract that cannot be inherited cannot
be inherited silently.

Four budget layers, in the order they are trusted, unchanged:

1. the **authorization artifact** — loaded before a pod exists, bound to the
   session's plan hash, and consulted before every priced decision;
2. the **soft stop** inside the driver — it refuses to start a stage it cannot
   finish before the artifact reserve;
3. this runner's teardown at the success marker or at any blocking failure,
   through the artifact gate;
4. the **independent watchdog**, which polls the provider, terminates at the hard
   threshold and verifies the pod is gone.

`--terminate-after` is set and never trusted; it has never been observed to fire.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..autoinit.authorization import AuthorizationError
from .artifact_gate import ArtifactManifest, evaluate_teardown, verify_extracted
from .log_relay import LogRelay, RelaySpec
from .provider import RunPodProvider, read_api_key
from .remote import JobSpec, SSHTarget, probe, start_detached
from .session import SessionContext, SessionSpec, missing_arguments

WS = "/workspace"
REPO = f"{WS}/aad"

PROBE_COMMAND = (
    "echo \"SETUP_DONE=$(grep -c 'MARKER:SETUP_DONE' {status} 2>/dev/null | tail -1)\"; "
    "echo \"HOST_COLD=$(grep -c 'HOST_COLD' {status} 2>/dev/null | tail -1)\"; "
    "echo \"SETUP_RC=$(grep -o 'SETUP_RC=[0-9]*' {log} 2>/dev/null | tail -1 | cut -d= -f2)\"; "
    "echo \"TAIL=$(tail -1 {log} 2>/dev/null)\""
)


def parse_setup_probe(stdout: str) -> dict:
    """Read the probe by LABEL, never by line position (see e8b: a $0.19 misread)."""
    out = {"setup_done": "0", "host_cold": "0", "setup_rc": "", "tail": ""}
    for line in stdout.splitlines():
        key, _, value = line.partition("=")
        if key.strip().lower() in out:
            out[key.strip().lower()] = value.strip()
    return out


class SessionRunner:
    """Runs one `SessionSpec`. Not a base class; there is nothing to override."""

    def __init__(self, spec: SessionSpec, args, repo_root: Path):
        self.spec = spec.validate()
        self.a = args
        self.repo_root = Path(repo_root)

        # The argument contract, checked at $0. Device-canary attempt 1 died at
        # $0.0603 because the machinery read three attributes its parser had
        # never defined, and nothing looked until a pod existed.
        missing = missing_arguments(args)
        if missing:
            raise AuthorizationError(
                f"{spec.session_id}: the argument namespace is missing "
                f"{missing}, which this runner reads. A session that cannot "
                "supply the runner's arguments must not create a pod.")

        self.scr = Path(args.scr)
        self.scr.mkdir(parents=True, exist_ok=True)
        self.auth = spec.authorization_loader(self.repo_root / spec.authorization_path)
        self.auth.require_plan(spec.plan_hash)
        # Before a pod can exist: the harness on disk must be the one this
        # authorization was granted against. An edited launcher, driver, setup
        # script or watchdog is an unrehearsed harness, and a paid run that
        # produces permanent artifacts must not be executed by one.
        self.harness = self.auth.require_harness(self.repo_root)
        self.key = os.environ.get("RUNPOD_API_KEY") or read_api_key(args.runpod_config)
        self.provider = RunPodProvider(self.key)
        self.cli = shutil.which("runpodctl") or os.path.expanduser(
            "~/.local/bin/runpodctl")
        if not Path(self.cli).is_file():
            raise SystemExit("runpodctl not found")

        self.ev: dict = {
            "schema": spec.schema,
            "session_id": spec.session_id,
            "session_spec": spec.as_dict(),
            "timeline": [], "stages": {},
            "authorization": self.auth.as_dict(),
            "session_plan_hash": spec.plan_hash,
            "harness_source_digest": self.harness["digest"],
            "harness_source_files": [f["path"] for f in self.harness["files"]],
        }
        self.ev.update(dict(spec.evidence_fields))
        self.pod_id = ""
        self.start_epoch = 0.0
        self.price = None
        self.plan = None
        self.endpoint = ("", "")
        self.image_digest = ""

    # -- helpers ----------------------------------------------------------
    def say(self, msg: str) -> None:
        line = f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}"
        print(line, flush=True)
        with open(self.scr / "launch.log", "a") as f:
            f.write(line + "\n")
        self.ev["timeline"].append(
            {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "elapsed_min": round(self.elapsed(), 2), "msg": msg})

    def elapsed(self) -> float:
        return (time.time() - self.start_epoch) / 60 if self.start_epoch else 0.0

    def usd(self) -> float:
        return self.elapsed() / 60 * (self.price or self.a.max_price)

    def save(self) -> None:
        (self.repo_root / self.a.out).write_text(
            json.dumps(self.ev, indent=2, default=str) + "\n")

    def context(self, **over) -> SessionContext:
        """What a spec callable may see. Never the runner itself."""
        host, port = self.endpoint
        ctx = SessionContext(
            scr=self.scr, args=self.a, auth=self.auth, evidence=self.ev,
            say=self.say, host=host,
            target=SSHTarget(host, port) if host else None,
            scp=(("scp", "-P", port, "-o", "StrictHostKeyChecking=no",
                  "-o", "UserKnownHostsFile=/dev/null") if host else ()),
            plan=self.plan, price=self.price, image_digest=self.image_digest,
            elapsed_minutes=self.elapsed(), spent_usd=self.usd())
        for k, v in over.items():
            setattr(ctx, k, v)
        return ctx

    # -- 1. plan and price -------------------------------------------------
    def make_plan(self) -> bool:
        try:
            self.plan = self.spec.budget.plan(
                price_per_hour=self.a.max_price,
                authorized_usd=self.auth.hard_cap_usd)
        except Exception as exc:                                  # noqa: BLE001
            self.say(f"ABORT: {type(exc).__name__}: {exc}")
            return False
        self.ev["budget_plan"] = self.plan.as_dict()
        try:
            self.auth.require_within_cap(self.plan.hard_terminate_usd,
                                         what="planned hard threshold")
            if hasattr(self.auth, "require_within_launch_limit"):
                self.auth.require_within_launch_limit(
                    self.plan.hard_terminate_usd, what="planned hard threshold")
        except AuthorizationError as exc:
            self.say(f"ABORT: {exc}")
            return False
        self.say(f"budget: expected {self.plan.expected_minutes:.0f} min "
                 f"${self.plan.expected_usd:.2f} · soft "
                 f"${self.plan.soft_stop_usd:.2f} · hard "
                 f"${self.plan.hard_terminate_usd:.2f} "
                 f"(authorized ${self.auth.hard_cap_usd:.2f})")
        return self.check_gpu_offered()

    def check_gpu_offered(self) -> bool:
        """Is the GPU offered at or below the priced rate?"""
        d = self.provider._gql(
            'query { gpuTypes(input:{id:"%s"}) { id securePrice '
            'lowestPrice(input:{gpuCount:1}) { stockStatus } } }' % self.a.gpu)
        rows = (d.get("data") or {}).get("gpuTypes") or []
        if not rows:
            self.say(f"ABORT: {self.a.gpu} not offered")
            return False
        self.price = rows[0].get("securePrice")
        self.ev["quoted_price_per_hour"] = self.price
        self.ev["stock_status"] = (rows[0].get("lowestPrice") or {}).get("stockStatus")
        self.say(f"{self.a.gpu} ${self.price}/h, stock {self.ev['stock_status']}")
        if self.price is None or self.price > self.a.max_price:
            self.say(f"ABORT: ${self.price}/h above the priced ${self.a.max_price}/h")
            return False
        return True

    # -- 2. precheck: everything this session reads, checked at $0 ---------
    def run_prechecks(self) -> bool:
        """The session's own gates, then the manifest's declared inputs.

        Every input a session reads is declared, so this is one loop rather than
        an overridden method per session — which is what stopped the continuation
        discovering an unstaged control after setup had been paid for.
        """
        ctx = self.context()
        for check in self.spec.precheck:
            ok, message = check(ctx)
            self.ev.setdefault("prechecks", []).append(
                {"check": getattr(check, "__name__", str(check)),
                 "ok": bool(ok), "message": message})
            if not ok:
                self.say(f"ABORT at $0: {message}")
                self.save()
                return False
            if message:
                self.say(message)

        need = [r.path for r in self.spec.setup.relay_inputs]
        missing: list[str] = []
        if need:
            try:
                from huggingface_hub import HfApi              # noqa: PLC0415
                present = set(HfApi().list_repo_files(
                    getattr(self.a, "relay_repo", "AlphaAvatar/aadistill-artifacts"),
                    repo_type="model"))
            except Exception as exc:                           # noqa: BLE001
                self.say(f"ABORT: cannot list the relay: {exc!r}"[:200])
                return False
            missing = [f for f in need if f not in present]
        local = [a.repo_path for a in self.spec.setup.local_assets]
        local_missing = [p for p in local if not (self.repo_root / p).exists()]
        self.ev["precheck"] = {"relay_needed": need, "relay_missing": missing,
                               "local_assets": local,
                               "local_missing": local_missing}
        if missing or local_missing:
            self.say(f"ABORT at $0: relay missing {missing}, "
                     f"local missing {local_missing}")
            self.save()
            return False
        self.say(f"precheck OK: {len(need)} relay inputs, {len(local)} local assets")
        return True

    # -- 3. create ---------------------------------------------------------
    def create(self) -> bool:
        deadline = (datetime.now(timezone.utc)
                    + timedelta(minutes=self.plan.hard_terminate_minutes))
        for attempt in range(1, self.a.create_attempts + 1):
            raw = subprocess.run(
                [self.cli, "pod", "create", "--image", self.a.image,
                 "--gpu-id", self.a.gpu, "--gpu-count", "1",
                 "--container-disk-in-gb", str(self.a.disk_gb), "--volume-in-gb", "0",
                 "--min-cuda-version", "13.0", "--ports", "22/tcp",
                 "--name", f"aadistill-{self.spec.session_id}",
                 "--terminate-after", deadline.strftime("%Y-%m-%dT%H:%M:%SZ")],
                capture_output=True, text=True, timeout=300)
            (self.scr / f"create_raw_{attempt}.txt").write_text(raw.stdout + raw.stderr)
            try:
                pid = json.loads(raw.stdout).get("id", "")
            except Exception:
                m = re.search(r'"id"\s*:\s*"([^"]+)"', raw.stdout + raw.stderr)
                pid = m.group(1) if m else ""
            if pid:
                try:
                    actual = json.loads(raw.stdout).get("costPerHr")
                    if actual is not None:
                        self.price = float(actual)
                except Exception:
                    pass
                if self.price > self.a.max_price:
                    self.say(f"ABORT: provisioned at ${self.price}/h — deleting")
                    subprocess.run([self.cli, "remove", "pod", pid],
                                   capture_output=True, timeout=120)
                    return False
                if not self.start_epoch:
                    self.start_epoch = time.time()
                    (self.scr / "pod_start_epoch").write_text(str(self.start_epoch))
                self.pod_id = pid
                (self.scr / "pod_id").write_text(pid)
                self.ev["pod_id"] = pid
                self.ev["actual_price_per_hour"] = self.price
                self.ev["terminate_after_utc"] = deadline.strftime("%Y-%m-%dT%H:%M:%SZ")
                self.say(f"created {pid} at ${self.price}/h")
                return True
            self.say(f"attempt {attempt}: create failed — "
                     f"{(raw.stdout + raw.stderr).strip()[:200]}")
            if attempt < self.a.create_attempts:
                time.sleep(self.a.create_retry_seconds)
        self.say("ABORT: could not create a pod")
        return False

    def launch_watchdog(self) -> Path:
        """Independent of the driver and of this runner, by construction."""
        journal = self.scr / "watchdog.jsonl"
        cmd = [sys.executable, str(self.repo_root / "scripts/pod/watchdog.py"),
               "--pod-id", self.pod_id,
               "--session-start-epoch", str(self.start_epoch),
               "--price-per-hour", str(self.price),
               "--hard-minutes", str(self.plan.hard_terminate_minutes),
               "--authorized-usd", str(self.auth.hard_cap_usd),
               "--journal", str(journal), "--poll-seconds", "60"]
        out = open(self.scr / "watchdog.out", "w")
        subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, cwd=self.repo_root,
                         env={**os.environ, "PYTHONPATH": str(self.repo_root / "src")},
                         start_new_session=True)
        self.say(f"watchdog detached — hard {self.plan.hard_terminate_minutes:.0f} min "
                 f"= ${self.plan.hard_terminate_usd:.2f}")
        return journal

    def wait_endpoint(self):
        deadline = time.time() + self.a.startup_limit_min * 60
        i = 0
        while time.time() < deadline:
            d = self.provider._gql(
                'query { pod(input:{podId:"%s"}) { runtime { ports '
                '{ ip publicPort privatePort type } } } }' % self.pod_id)
            rt = ((d.get("data") or {}).get("pod") or {}).get("runtime")
            for p in (rt or {}).get("ports") or []:
                if p.get("privatePort") == 22 and p.get("type") == "tcp":
                    self.say(f"TCP 22 at {p['ip']}:{p['publicPort']} after "
                             f"{self.elapsed():.1f} min")
                    return str(p["ip"]), str(p["publicPort"])
            i += 1
            if i % 6 == 0:
                self.say(f"  starting ({i * 10}s) — ${self.usd():.2f}")
            time.sleep(10)
        return None

    def read_image_digest(self, target: SSHTarget) -> str:
        """The image the pod is really running, from the provider, not the arg."""
        d = self.provider._gql(
            'query { pod(input:{podId:"%s"}) { imageName machine { podHostId } } }'
            % self.pod_id)
        pod = ((d.get("data") or {}).get("pod") or {})
        name = pod.get("imageName") or self.a.image
        rc = target.run(
            "cat /etc/podinfo/image_digest 2>/dev/null || "
            "nvidia-smi --query-gpu=driver_version --format=csv,noheader", timeout=60)
        extra = (rc.stdout or "").strip().splitlines()[:1]
        digest = f"{name}@{extra[0]}" if extra else name
        self.ev["image_identity"] = {"image_name": name, "observed": extra,
                                     "digest": digest}
        return digest

    # -- 4. setup ----------------------------------------------------------
    def setup_on_draw(self, draw: int) -> str:
        ep = self.wait_endpoint()
        if not ep:
            return "no_endpoint"
        host, port = ep
        self.endpoint = (host, port)
        target = SSHTarget(host, port)
        for _ in range(30):
            if target.run("true", timeout=30).returncode == 0:
                break
            time.sleep(10)
        scp = ["scp", "-P", port, "-o", "StrictHostKeyChecking=no",
               "-o", "UserKnownHostsFile=/dev/null"]
        self.say(f"draw {draw}: ssh reachable — ${self.usd():.2f}")

        token = Path(self.a.token_src)
        if not token.is_file() or token.stat().st_size == 0:
            return "no_hf_token"
        target.run(f"mkdir -p {WS}/hf {WS}/assets && chmod 700 {WS}/hf", timeout=60)
        subprocess.run(scp + [str(token), f"root@{host}:{WS}/hf/token"],
                       capture_output=True, timeout=180)
        if target.run(f"test -s {WS}/hf/token", timeout=60).returncode != 0:
            return "empty_hf_token"
        # Only what the manifest declares. The uplink is 0.72 MB/s, so the
        # frozen search assets are ~3 s and need no relay round trip — but a
        # session that declares none now sends none, and setup copies none.
        for asset in self.spec.setup.local_assets:
            subprocess.run(scp + ["-r", str(self.repo_root / asset.repo_path),
                                  f"root@{host}:{WS}/assets/{asset.dest_name}"],
                           capture_output=True, timeout=600)
        subprocess.run(scp + [str(self.repo_root
                                  / "scripts/pod/autoinit_preflight_setup.sh"),
                              f"root@{host}:{WS}/"], capture_output=True, timeout=180)

        self.image_digest = self.read_image_digest(target)
        self.say(f"draw {draw}: image identity {self.image_digest}")
        self.say(f"draw {draw}: running setup")
        env = self.spec.setup_environment(session_commit=self.a.session_commit,
                                          bundle=self.a.bundle)
        self.ev["setup_environment"] = {k: v for k, v in sorted(env.items())}
        rendered = " ".join(f"{k}={_shell_quote(v)}" for k, v in sorted(env.items()))
        target.run(
            f"cd {WS} && {rendered} "
            f"bash {WS}/autoinit_preflight_setup.sh > {WS}/setup.log 2>&1; "
            f"echo SETUP_RC=$? >> {WS}/setup.log",
            timeout=self.a.setup_timeout_s)
        result = parse_setup_probe(target.run(
            PROBE_COMMAND.format(status=self.spec.status_path,
                                 log=f"{WS}/setup.log"),
            timeout=120).stdout)
        self.ev["stages"].setdefault("setup", []).append({"draw": draw, **result})
        if result["host_cold"] not in ("", "0") or result["setup_rc"] == "90":
            return "cold"
        if result["setup_done"] in ("", "0"):
            tail = target.run(f"tail -40 {WS}/setup.log", timeout=120).stdout
            self.say(f"setup did not reach SETUP_DONE:\n{tail[-2000:]}")
            return "setup_failed"
        self.say(f"draw {draw}: setup complete — ${self.usd():.2f}")
        return "ok"

    # -- 5. run ------------------------------------------------------------
    def run(self) -> bool:
        if not self.make_plan() or not self.run_prechecks():
            return False
        for draw in range(1, self.a.host_draws + 1):
            if not self.create():
                return False
            self.ev.setdefault("watchdog_journals", []).append(
                str(self.launch_watchdog()))
            self.save()
            outcome = self.setup_on_draw(draw)
            if outcome == "ok":
                break
            if outcome in ("cold", "no_endpoint") and draw < self.a.host_draws:
                self.say(f"{outcome.upper()} on draw {draw} — abandoning "
                         f"{self.pod_id} and redrawing")
                subprocess.run([self.cli, "remove", "pod", self.pod_id],
                               capture_output=True, timeout=180)
                self.pod_id = ""
                continue
            self.say(f"ABORT after draw {draw}: {outcome}")
            if self.pod_id:
                self.teardown_now(f"setup {outcome}")
            return False
        else:
            self.say(f"ABORT: {self.a.host_draws} consecutive unusable draws")
            return False

        host, port = self.endpoint
        target = SSHTarget(host, port)
        scp = ["scp", "-P", port, "-o", "StrictHostKeyChecking=no",
               "-o", "UserKnownHostsFile=/dev/null"]

        # Inputs the driver needs but does not fetch itself.
        if not self.spec.materialize_inputs(self.context()):
            self.teardown_now("inputs did not materialize")
            return False

        job = start_detached(target, JobSpec(
            job_id=self.spec.driver_job_id, workdir=REPO,
            command=self.spec.driver_command(self.context(), self.plan),
            job_dir=f"{WS}/jobs", log_path=self.spec.run_log_path,
            status_path=self.spec.status_path,
            env={"PYTHONPATH": f"{REPO}/src"}),
            start_timeout=120, verify_timeout=60)
        self.ev["driver_job"] = job.as_dict()
        self.say(f"driver detached, pid {job.pid}, confirmed by {job.confirmed_by} "
                 f"— ${self.usd():.2f}")
        self.save()

        specs = [
            RelaySpec(self.spec.run_log_path,
                      Path(self.spec.run_log_path).name, required=False),
            RelaySpec(self.spec.status_path,
                      Path(self.spec.status_path).name, required=False),
            RelaySpec(f"{REPO}/artifacts/audit/{self.spec.artifacts.audit_dirname}/"
                      f"{self.spec.artifacts.evidence_filename}",
                      self.spec.artifacts.evidence_filename, required=False),
        ]
        specs += [RelaySpec(remote, local, required=False) for remote, local
                  in self.spec.artifacts.extra_relay_streams(self.context())]
        relay = LogRelay(target, tuple(specs), self.scr / "relay")

        last, terminal = "", ""
        deadline = time.time() + self.a.poll_limit_min * 60
        while time.time() < deadline:
            time.sleep(self.a.poll_seconds)
            r = relay.sync_once()
            if r.errors:
                self.say(f"  relay errors: {list(r.errors)[:2]}")
            st = target.run(f"tail -1 {self.spec.status_path} 2>/dev/null",
                            timeout=60).stdout.strip()
            if st and st != last:
                last = st
                self.say(f"  {st} — ${self.usd():.2f}")
            if f"MARKER:{self.spec.markers.success}" in st:
                terminal = self.spec.markers.success
                break
            hit = [m for m in self.spec.markers.failure if f"MARKER:{m}" in st]
            if hit:
                terminal = hit[0]
                self.say(self.spec.markers.failure_note)
                break
            state = self.provider.get(self.pod_id)
            if not state.billing:
                terminal = "POD_GONE"
                self.say("the pod is gone — the watchdog acted")
                break
            live, _ = probe(target, job)
            if live != "ALIVE":
                terminal = f"DRIVER_{live}"
                self.say(f"driver no longer running: {live}")
                break
        self.ev["terminal"] = terminal or "POLL_LIMIT"
        self.say(f"polling ended: {self.ev['terminal']} — ${self.usd():.2f}")
        relay.sync_once()
        self.save()

        if terminal == "POD_GONE":
            self.finish_emergency()
            return False
        return self.collect_and_teardown(target, host, scp, terminal)

    # -- 6. collect --------------------------------------------------------
    def collect_and_teardown(self, target, host, scp, terminal: str) -> bool:
        art = self.spec.artifacts
        success = self.spec.markers.success
        cc = (f"cd {REPO} && PYTHONPATH={REPO}/src /opt/train/bin/python "
              "scripts/pod/collect_artifacts.py")
        audit = f"{REPO}/artifacts/audit/{art.audit_dirname}"
        target.run(f"mkdir -p {audit}/session && "
                   f"cp {self.spec.run_log_path} {self.spec.status_path} "
                   f"{WS}/setup.log {audit}/session/ 2>/dev/null "
                   "|| true", timeout=120)
        # A blocking failure has a smaller required set: the products do not
        # exist, and demanding them would block teardown on artifacts the run
        # correctly refused to produce.
        spec_path = art.spec_failed if terminal != success else art.spec_success
        man, arc = f"{WS}/manifest.json", f"{WS}/{art.archive_basename}"
        r_man = target.run(
            f"{cc} manifest --root {REPO}/artifacts --spec {REPO}/{spec_path} "
            f"--out {man} --settle-seconds {self.a.settle_seconds}", timeout=900)
        self.say(f"  manifest rc={r_man.returncode}\n{r_man.stdout.strip()[-900:]}")
        r_arc = target.run(f"{cc} archive --manifest {man} --out {arc}", timeout=1800)
        r_ver = target.run(f"{cc} verify-archive --manifest {man} --archive {arc}",
                           timeout=900)
        store = self.scr / "store"
        store.mkdir(exist_ok=True)
        for remote, local in ((man, store / "manifest.json"),
                              (arc, store / art.archive_basename)):
            subprocess.run(scp + [f"root@{host}:{remote}", str(local)],
                           capture_output=True, timeout=1800)

        local_ok, manifest = False, None
        if (store / "manifest.json").is_file():
            import tarfile
            extract = store / "extracted"
            extract.mkdir(exist_ok=True)
            try:
                with tarfile.open(store / art.archive_basename) as tar:
                    tar.extractall(extract, filter="data")
                manifest = ArtifactManifest.load(store / "manifest.json")
                problems = verify_extracted(extract, manifest)
                local_ok = not problems
                self.ev["local_hash_problems"] = problems
            except Exception as exc:                              # noqa: BLE001
                self.ev["local_verify_error"] = f"{type(exc).__name__}: {exc}"

        # Products are fetched whenever they EXIST, which is whenever the
        # blocking stages passed — not only when the whole session succeeded.
        # This was `if terminal == "ALL_DONE"` on 2026-08-13, and it destroyed
        # both controls of a $2.82 session.
        stage2_passed = self.spec.markers.stage2_passed(
            terminal, self.ev.get("driver_stages") or {})
        # The reports are fetched BEFORE the products: Phase A's
        # `fetch_products` reads `leaf_retention.json` to decide which
        # initializations come home.
        for name in art.report_names:
            subprocess.run(scp + [f"root@{host}:{audit}/{name}", str(store / name)],
                           capture_output=True, timeout=600)
            if (store / name).is_file():
                self.ev.setdefault("fetched_reports", []).append(name)
        fetched = art.fetch_products(
            self.context(host=host, target=target, scp=tuple(scp),
                         stage2_passed=stage2_passed))
        self.ev["checkpoints_fetched"] = fetched
        # Did this session secure what it OWES off-pod? Asked separately from
        # `checkpoint_hashes_matched`, which is `all([])` and therefore vacuously
        # true when the fetch returned nothing at all.
        secured_ok, secured_why = art.products_secured(
            self.context(host=host, target=target, scp=tuple(scp),
                         stage2_passed=stage2_passed), fetched)
        self.ev["required_products_secured"] = {"ok": bool(secured_ok),
                                                "why": secured_why}
        if not secured_ok:
            self.say(f"  PRODUCTS NOT SECURED: {secured_why}")

        done = terminal == success
        state = {
            "training_complete": done,
            "evaluation_complete": done,
            "artifact_manifest_created": (store / "manifest.json").is_file(),
            "required_files_present": bool(manifest and manifest.ok),
            "final_streams_quiescent": bool(manifest and manifest.final_streams_quiescent),
            "archive_created": r_arc.returncode == 0,
            "archive_contents_verified": r_ver.returncode == 0,
            "transfer_complete": (store / art.archive_basename).is_file(),
            "local_hashes_verified": local_ok,
            "checkpoint_hashes_matched": all(f.get("rc") == 0 for f in fetched),
            "report_inputs_verified": local_ok,
            "required_products_secured": bool(secured_ok),
        }
        decision = evaluate_teardown(
            state,
            emergency_budget=not done,
            emergency_reason=(
                "" if done else
                f"a blocking stage failed ({terminal}); the products of the "
                "stages that did not run do not exist and must not be demanded. "
                "Evidence is collected under the reduced spec and the pod is "
                "torn down."),
            incomplete_event_streams=() if done
            else art.event_streams(self.context()),
            # The manifest's evidence about WHY quiescence failed, so the gate
            # can tell a truncated stream from an artifact that was never
            # written. A session declaring no streams and missing its one report
            # is the second, and must not be asked to name streams it has none
            # of. `None` when there is no manifest: no evidence, strict rule.
            streams_at_risk=(
                tuple(manifest.completion_marker_failures)
                + tuple(manifest.still_being_written)) if manifest else None)
        self.ev["teardown_gate"] = decision.as_dict()
        self.ev["manifest_summary"] = (
            {"ok": manifest.ok, "entries": len(manifest.entries),
             "missing": manifest.missing,
             "still_being_written": manifest.still_being_written}
            if manifest else None)
        self.say(f"teardown gate: allowed={decision.allowed} "
                 f"failed={decision.failed_check}")
        self.save()
        if not decision.allowed:
            self.say("GATE BLOCKED — the runner is NOT deleting the pod; the "
                     "watchdog remains the backstop")
            return False
        self.teardown_now("gate passed" if done else f"gate passed after {terminal}")
        return done

    def finish_emergency(self) -> None:
        streams = self.spec.artifacts.event_streams(self.context())
        events = {}
        for remote in streams:
            name = Path(remote).parent.name
            p = self.scr / "relay" / f"{name}.train_log.jsonl"
            events[name] = sum(1 for _ in p.open()) if p.is_file() else 0
        decision = evaluate_teardown(
            {"training_complete": False, "evaluation_complete": False,
             "artifact_manifest_created": False, "required_files_present": False,
             "final_streams_quiescent": False},
            emergency_budget=True,
            emergency_reason=(f"the watchdog terminated the pod at the hard "
                              f"threshold ({self.plan.hard_terminate_minutes:.0f} "
                              f"min / ${self.plan.hard_terminate_usd:.2f})"),
            incomplete_event_streams=streams)
        self.ev["teardown_gate"] = decision.as_dict()
        self.ev["relayed_events"] = events
        self.say(f"EMERGENCY: only relayed snapshots survive — {events}")
        self.save()

    def teardown_now(self, why: str) -> None:
        self.say(f"deleting pod ({why})")
        subprocess.run([self.cli, "remove", "pod", self.pod_id],
                       capture_output=True, timeout=180)
        st = None
        if self.spec.teardown.require_provider_confirmation:
            for _ in range(18):
                time.sleep(10)
                st = self.provider.get(self.pod_id)
                if not st.billing:
                    break
        self.ev["final_pod_state"] = {
            "exists": st.exists if st else None,
            "desired_status": st.desired_status if st else None,
            "billing": st.billing if st else None}
        self.ev["provider_confirms_gone"] = bool(st and not st.billing)
        self.ev["cost"] = {"price_per_hour": self.price,
                           "elapsed_minutes": round(self.elapsed(), 2),
                           "actual_usd": round(self.usd(), 4),
                           "authorized_usd": self.auth.hard_cap_usd,
                           "within_authorization": self.usd() <= self.auth.hard_cap_usd}
        self.say(f"pod deleted — {self.elapsed():.1f} min, ${self.usd():.2f}; "
                 f"provider confirms gone: {self.ev['provider_confirms_gone']}")
        self.save()


def _shell_quote(value: str) -> str:
    """Quote a setup environment value. A manifest value is data, not source."""
    if value == "" or re.search(r"[^\w@%+=:,./-]", value):
        return "'" + value.replace("'", "'\"'\"'") + "'"
    return value


def run_session(spec: SessionSpec, args, repo_root: Path, *,
                summary: str = "") -> int:
    """Construct, run and record one session. The launcher's whole `main`.

    Identical in shape to every launcher's `main`, so no session can forget to
    record `passed`, to write the record, or to tear down after a runner error.
    """
    runner = SessionRunner(spec, args, repo_root)
    ok = False
    try:
        ok = runner.run()
    except Exception as exc:                                      # noqa: BLE001
        runner.ev["launcher_error"] = f"{type(exc).__name__}: {exc}"
        runner.say(f"LAUNCHER ERROR: {type(exc).__name__}: {exc}")
        if runner.pod_id:
            runner.teardown_now("launcher error")
    runner.ev["passed"] = bool(ok)
    runner.ev["cleanup_is_not_success"] = (
        "artifacts are collected and the pod is torn down on every path; the "
        "session outcome is decided by the driver's terminal marker alone")
    runner.ev["finished_utc"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds")
    runner.save()
    print(f"\n{spec.session_id} {'COMPLETE' if ok else 'INCOMPLETE'} — "
          f"{repo_root / args.out}. {summary}")
    return 0 if ok else 11
