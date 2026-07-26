# 2026-07-26 — Infrastructure incident: RunPod "pods never start" was a misdiagnosis

- **Agent:** Claude Code (Opus 5), during the `s2_blocks_v1` resume session.
- **Type:** infrastructure incident + root-cause analysis. No model training
  involved; no experiment result is affected.
- **Cost of the incident:** ~$0.80 wasted on 2026-07-26 (three pods deleted
  while healthy) + ~$0.15 in this session (a fourth healthy pod deleted for
  the same reason) ≈ **$0.95 total**, plus one paused work session.

## Symptom

`runpodctl pod create` succeeds, the pod reports `desiredStatus: RUNNING`,
but:

- `runpodctl pod get <id>` shows `uptimeSeconds: 0` indefinitely;
- `runpodctl ssh info <id>` returns `{"error": "pod not ready"}` forever.

This was read as "the host never started the container, billing runs
anyway", and the response was to delete after ~8 min and re-allocate on a
different GPU pool. Repeating that produced an unbroken run of "failures"
across two GPU types (L40S, RTX 6000 Ada), two images (the playbook
`runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404` and the official
`runpod-torch-v240` template), and two regions (NL, US).

## Root cause — two independent broken signals, no broken pod

**The pods were running normally the entire time.** Both indicators used to
judge readiness are unreliable, and they fail together, which made a
healthy pod look dead.

### 1. `runpodctl` 2.7.1 reports `uptimeSeconds: 0` for healthy pods

The field is not populated by the CLI. The value exists and is correct in
the GraphQL API the CLI wraps. Measured on pod `mhak4avc7kt7cz` at the same
moment:

| Source | Value |
| --- | --- |
| `runpodctl pod get` → `uptimeSeconds` | `0` |
| GraphQL `pod.runtime.uptimeInSeconds` | `509`, then `658` |
| Actual `ssh` connection | RunPod login banner, `exit=0` |

Restarting the pod reset the GraphQL counter to `15` and it climbed again,
confirming it is a live value while the CLI copy stayed pinned at `0`.

### 2. `ssh info: "pod not ready"` means "no TCP 22 mapping", not "pod dead"

The staged create command passed no `--ports`, so the pod came up with only
an HTTP mapping (`19123 → 60481`, `isIpPublic: false`). With no public TCP
22 endpoint there is nothing for `ssh info` to report, so it returns
`"pod not ready"` — permanently, on a perfectly healthy pod.

The `ssh.runpod.io` **proxy** route worked the whole time and authenticated
with both keys on file. (The proxy needs a PTY: without `-tt` it answers
`Error: Your SSH client doesn't support PTY`. That error is itself proof the
pod is reachable. The proxy does not support `scp`/`sftp`, so a direct TCP
22 mapping is still required for this project's transfer step.)

### Why it worked on 2026-07-22 and 2026-07-25

Those sessions used the same create command with no `--ports` and got a
working `ssh info` with a host and port. Port 22 was evidently exposed by
default before 2026-07-26 and is not now — a RunPod-side default change or
a `runpodctl` upgrade. **The variable was never the GPU, the image, the
region, or host capacity**, which is why every escalation along those axes
changed nothing.

## Corrected procedure

1. **Always create with `--ports "22/tcp,8888/http"`.**
2. **Never use `runpodctl pod get`'s `uptimeSeconds` to judge readiness.**
   Use GraphQL:

   ```bash
   KEY=$(python3 -c "import re;t=open('$HOME/.runpod/config.toml').read();\
   print(re.search(r'apikey\s*=\s*(.+)',t,re.I).group(1).strip().strip('\"').strip(\"'\"))")
   curl -s -X POST "https://api.runpod.io/graphql?api_key=$KEY" \
     -H 'Content-Type: application/json' \
     -d '{"query":"query { pod(input:{podId:\"<id>\"}) { runtime { uptimeInSeconds ports { ip publicPort privatePort type } } } }"}'
   ```

   `runtime: null` = container genuinely still starting. A non-null
   `uptimeInSeconds` = the container is up. Poll until a `tcp`/`privatePort
   22` entry appears, then use that `ip:publicPort` for ssh/scp.

   **Config-parsing gotcha:** `~/.runpod/config.toml` values are
   **single-quoted** (`apikey = 'xxx'`, key name lowercase). Strip `'` as
   well as `"`, or the key is malformed and GraphQL returns `{"error":{}}`
   with no explanation.

3. **A real SSH connection is the only ground truth.** Before declaring a
   pod dead, try to connect. An 8-minute "zero uptime" rule based on the
   CLI field must not be used again.
4. **Recovering a pod created without ports** (no re-allocation needed):
   `runpodctl pod update <id> --ports "22/tcp,8888/http"` then
   `runpodctl pod restart <id>`; the TCP mapping appears ~30 s later.

## Verification that the fix works

- Pod `mhak4avc7kt7cz` (RTX 6000 Ada), declared "stalled" by the old rule,
  was recovered in place with `pod update --ports` + `restart`. SSH endpoint
  appeared within 30 s; `nvidia-smi` reported RTX 6000 Ada 49140 MiB, driver
  550.127.08; `/workspace` 120 GB local NVMe.
- The full `setup.sh` then passed all four markers on that pod
  (`ENV_READY` → `CKPT_READY` → `TESTS_PASSED` → `SETUP_DONE`), including
  transfer/checkpoint sha256 verification, cu128 torch on driver 550, and
  the pod-side pytest suite. So the "unusable" hardware was fully working.
- That pod was then discarded anyway, by maintainer instruction, to move the
  run back to **L40S** for hardware comparability with the 2026-07-22 s1 run
  and the 2026-07-25 A/B (both L40S). Discarding it was a deliberate
  experimental-control choice, not a hardware failure.

## Consequences for prior logs

`logs/STATE.md` and `scripts/pod/AGENTS.md` described the 2026-07-26 event as
a RunPod provisioning failure ("host never started the container"). That
description is **wrong** and has been corrected. No experiment result,
checkpoint, or metric is affected — nothing had started training when the
misdiagnosis occurred.

## Lesson (P11)

Two status fields agreeing does not make them independent evidence: both
`uptimeSeconds` and `ssh info` derive from the same missing runtime/port
data, so they failed as a pair and looked like corroboration. When a cheap
end-to-end test exists — here, one SSH attempt costing seconds — prefer it
over any status field, especially before spending money on re-allocation.
