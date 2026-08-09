# scripts/pod — GPU pod session scripts

**Status (2026-07-27): parameterized for multi-arm sessions.** The scripts no
longer hardcode a run; the session is declared in `run_env.sh` and everything
else reads it. The previous single-run versions (`orchestrate_s2v1.sh`,
`verify_and_report_s2v1.py`) are in git history at commit `f74e5ed`.

---

## Session contract (REQUIRED for every session after 2026-08-09)

E6b overran its authorization by $0.56 and lost both arms' machine-readable
training event streams. Four rules follow, and a new launcher that does not
implement them is not compliant. Full account:
[`logs/e6b_protocol_deviations.md`](../../logs/e6b_protocol_deviations.md);
decision record 2026-08-09.

**1. Never depend on the driver-start ssh returning.** E6b's launcher used the
correct `setsid nohup … < /dev/null & disown` form and blocked for 434 minutes
anyway — byte-identical to E6's, which returned in 74 s. Start the driver with:

```bash
JOB=$(python3 scripts/pod/start_job.py --host "$HOST" --port "$PORT" \
        --job-id e7_driver --workdir /workspace/aad \
        --log /workspace/e7_run.log --status /workspace/e7.status \
        --env TEACHER_REVISION="$TEACHER_REVISION" \
        --command "/opt/train/bin/python scripts/pod/e7_driver.py --stage all")
```

It returns within `start_timeout + verify` whether or not the channel closes,
prints a durable job descriptor (pid, log, status, marker paths), and exits 3 —
*do not proceed to poll* — if the job cannot be confirmed running.

**2. Run the watchdog beside the launcher, detached from it.** Immediately after
the pod is created:

```bash
setsid nohup python3 scripts/pod/watchdog.py \
  --pod-id "$POD_ID" --session-start-epoch "$(cat "$SCR/pod_start_epoch")" \
  --price-per-hour 0.99 --hard-minutes "$HARD_MINUTES" \
  --authorized-usd "$AUTHORIZED_USD" --journal "$SCR/watchdog.jsonl" \
  > "$SCR/watchdog.out" 2>&1 < /dev/null &
```

It touches only the provider control plane, so blocked SSH, a hung driver, a
crashed trainer and a failed collection are all invisible to it and all
irrelevant. **`--terminate-after` is a redundant third layer**: it has never been
observed to fire and does not count as a stop mechanism. Keep setting it; never
rely on it.

**3. Poll the pod, not the log.** A blocked launcher writes no lines, and
silence then looks exactly like an idle session — that is how seven hours and
~$4 went unnoticed. `watchdog.SessionWatcher.assess` requires provider state and
has no verdict named `IDLE`.

**4. Collect artifacts from a declared manifest, and gate teardown on it.**

```bash
python3 scripts/pod/collect_artifacts.py manifest --root artifacts \
  --spec configs/<session>/artifacts.json --out /workspace/manifest.json
python3 scripts/pod/collect_artifacts.py archive --manifest … --out …
python3 scripts/pod/collect_artifacts.py verify-archive --manifest … --archive …
# … transfer …
python3 scripts/pod/collect_artifacts.py verify-local --manifest … --root "$STORE"
python3 scripts/pod/collect_artifacts.py gate --state "$SCR/gate.json"
```

The spec **must** declare `train_log.jsonl` and `run_manifest.json` as required
for every training arm. A missing required artifact blocks teardown. Only the
cost watchdog may override, and it must record the reason and what was lost.

**Banned:** `$(ls …)` / backticks inside a quoted ssh command. A pattern that
matches nothing yields an empty substitution and a silently shorter archive that
still exits 0 and still hashes. Lint: `tests/pod/test_operational_hardening.py`.
`e3/e4/e5_launch.sh` are exempted by name as frozen records.

**Budget:** four thresholds, not one. Use
`aadistill.infrastructure.budget.plan_session` — expected / soft stop /
artifact-recovery reserve / hard terminate — and pass `hard_terminate_minutes`
to the watchdog. **Price the step at 4.15 s (E6b measured), never at E4's
3.625 s**; the planner refuses the superseded figure by name. Name evaluation
and checkpointing as their own phases: wall clock per step was 4.21 s because
they are not free.

**Durability:** mirror `train_log.jsonl` off the pod continuously with
`aadistill.infrastructure.log_relay.LogRelay` from the polling loop. A structured
event stream may not exist only inside an ephemeral pod until teardown.

## Files

| file | role |
| --- | --- |
| `start_job.py` | **required**: start the remote driver detached; prints a durable job descriptor; exit 3 = not confirmed, do not poll |
| `watchdog.py` | **required**: independent provider-level cost backstop; polls, terminates, verifies the pod is gone, journals every attempt. `--simulate` rehearses thresholds with no pod |
| `collect_artifacts.py` | **required**: manifest / archive / verify-archive / verify-local / gate — manifest-driven collection and the ordered teardown gate |
| `reconstruct_training_events.py` | parses a driver console log into a **derived** event stream with explicit provenance; never a replacement for `train_log.jsonl` |
| `run_env.sh` | **the only file to edit per session**: arms (name/config/step tag), reference checkpoints + revisions, transfer artifact names, eval settings |
| `setup.sh` | env + repo bundle + data + cu128 torch + checkpoints + tests; markers `ENV_READY`→`CKPT_READY`→`TESTS_PASSED`→`SETUP_DONE` |
| `score_refs.sh` | scores reference checkpoints on `eval_behavior_v0` **before training**, so every arm is comparable on one device and eval bugs surface in minutes, not after hours of training; marker `REFS_SCORED` |
| `train.sh` | `train.sh <RUN_NAME> <CONFIG> [--resume]`; markers `TRAIN_DONE:<run>` / `TRAIN_FAILED:<run>` |
| `post_run.sh` | `post_run.sh <RUN_NAME> <CONFIG> <STEP_TAG>`; gate evals (bf16 holdout, INT8 both scopes, `eval_behavior_v0`, gen smoke) + hashes + HF upload; marker `POST_DONE:<run>` |
| `orchestrate.sh` | dev-box driver: `POD_ID=… HOST=… PORT=… bash scripts/pod/orchestrate.sh`; loops arms, fetches, verifies, reports, commits, tears down |
| `verify_and_report.py` | `verify --run <name>` (independent HF upload check) and `report --run a,b` (multi-arm write-up + mechanical decision rules) |
| `hashes_transfer.txt` / `hashes_ckpt.txt` | sha256 manifests, re-verified pod-side by `setup.sh` and again by the orchestrator |

**Markers are arm-scoped** (`TRAIN_DONE:<run>`), which is what makes a multi-arm
session safe to leave unattended. Keep that protocol and the sha256 steps exactly
as they are.

## Session procedure

1. ```bash
   runpodctl pod create --image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 \
     --gpu-id "NVIDIA L40S" --container-disk-in-gb 150 --volume-in-gb 0 \
     --ports "22/tcp,8888/http" --name aadistill-<session> \
     --terminate-after "$(date -u -d '+8 hours' +%Y-%m-%dT%H:%M:%SZ)"
   ```

   **`--terminate-after` takes an absolute UTC datetime**
   (`2026-07-27T18:04:14Z`), *not* a duration like `+8h` — a duration is
   rejected, which would leave the session without its cost backstop. Size the
   container disk for the whole session: a two-arm run used ~30 GB with three
   staged checkpoints plus rolling checkpoints, so 150 GB is comfortable and
   costs cents.

   **`--ports "22/tcp"` is mandatory** (see step 2). Prefer **L40S** so
   throughput/memory stay comparable to s1, the A/B and `s2_blocks_v1`. RTX 6000
   Ada (48 GB) is a verified fallback — note the change in the experiment log if
   used. Prefer pod-local disk (`--volume-in-gb 0`): a `/workspace` network
   volume may be MooseFS, which serves stale reads after write bursts.

2. Wait for readiness. **Do NOT trust `runpodctl pod get`'s `uptimeSeconds`: in
   runpodctl 2.7.1 it is always `0`, even on a healthy running pod.** And
   **`runpodctl ssh info` returning `"pod not ready"` does not mean the pod
   failed** — it means no public TCP 22 mapping exists, which never appears if
   the pod was created without `--ports "22/tcp"`. Both were verified against the
   API on 2026-07-26; together they cost ~$0.95 in healthy pods deleted as
   "stuck" (`logs/EXPERIMENTS.md`).

   Use GraphQL, and treat an actual SSH connection as the only ground truth:

   ```bash
   KEY=$(python3 -c "import re;t=open('$HOME/.runpod/config.toml').read();\
   print(re.search(r'apikey\s*=\s*(.+)',t,re.I).group(1).strip().strip('\"').strip(\"'\"))")
   curl -s -X POST "https://api.runpod.io/graphql?api_key=$KEY" \
     -H 'Content-Type: application/json' \
     -d '{"query":"query { pod(input:{podId:\"<id>\"}) { runtime { uptimeInSeconds ports { ip publicPort privatePort type } } } }"}'
   ```

   `runtime: null` = still starting. Poll until a `tcp`/`privatePort 22` entry
   appears and use that `ip:publicPort`. Config values are **single-quoted** and
   the key is lowercase `apikey` — strip `'` as well as `"`, or GraphQL returns
   `{"error":{}}`. Recover a portless pod in place with
   `runpodctl pod update <id> --ports "22/tcp,8888/http"` + `pod restart`.

3. `scp` to `/workspace/`: `run_env.sh`, `setup.sh`, `score_refs.sh`, `train.sh`,
   `post_run.sh`, `hashes_transfer.txt`, `hashes_ckpt.txt`, plus the HF token
   (from `hf auth token`) to `/workspace/hf/token`, mode 600. Never echo the
   token into logs.
4. `bash /workspace/setup.sh`, then run the orchestrator from the dev box under
   tmux/nohup. It drives everything else and deletes the pod only after upload
   verification passes.

## Teacher-generation sessions (vLLM) — different image, different rules

The scripts above are the **training** path. The 2026-07-30 gate and the
2026-08-01 corpus build ran a generation job instead, under its own venv, and
paid for four infrastructure lessons (`logs/EXPERIMENTS.md` §9):

- **Create the pod with `--min-cuda-version 13.0`.** vLLM 0.26.0's wheel links
  `libcudart.so.13`; a 570.x-driver host cannot run it, and
  `--torch-backend=cu128` does not help because it changes torch, not the vLLM
  extension.
- **`ninja` must be on `PATH`.** FlashInfer JIT-builds the top-k sampling kernel
  during warmup, and `top_k=20` is in the official preset — without ninja the
  engine dies *after* loading weights.
- **Put the venv on the container disk**, not `/workspace`: a torch install onto
  the network mount took >9 minutes.
- **`tar` needs `--no-same-owner`** on that mount or it exits non-zero on chown.

The build itself is one long unattended command; drive it from the dev box under
tmux/nohup with the same marker protocol (`MARKER:GEN_DONE`,
`MARKER:LADDER_DONE`, `MARKER:GATE_EXIT:<rc>`, `MARKER:ALL_DONE`).

## Cost discipline

**Tear the pod down when the job finishes, not when `--terminate-after` fires.**
The corpus build finished at 06:27 and the pod was deleted at 15:14, because
polling was driven by user prompts and the backstop was 32 h — **~$8.70 of the
$25.56 run was idle time**. Tie teardown to the completion marker; the
`--terminate-after` timestamp is a backstop, never the plan.

**Fix the bundle so `code_state` carries a git commit.** The corpus v2 manifest
records `code_state_error` instead of a commit, because the bundle was unpacked
outside a git checkout (P4 gap, `logs/EXPERIMENTS.md` §10). Ship the commit hash
with the bundle, or unpack into a real checkout.

## Before each session

- **Regenerate the git bundle** at the current commit and re-upload it, then
  update `hashes_transfer.txt` — `setup.sh` verifies hashes and fails loudly on
  a stale bundle. The tracked `data/eval_behavior_v0/prompts.jsonl` travels with
  the bundle, so a stale bundle also means a stale prompt set.
- **Stage any start checkpoint not already on the relay** and add its per-file
  sha256 to `hashes_ckpt.txt` (dev→HF upload measured ~680 KB/s; HF→pod is fast).
- **Amortize setup**: it was 38 min (19%) of the last session's spend. Multiple
  arms on one pod pay it once.
- **Keep seed `20260726`** in every config that must be comparable with the
  completed runs (decision record 2026-07-27).
