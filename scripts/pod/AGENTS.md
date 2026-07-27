# scripts/pod — GPU pod session scripts

**Status (2026-07-27): parameterized for multi-arm sessions.** The scripts no
longer hardcode a run; the session is declared in `run_env.sh` and everything
else reads it. The previous single-run versions (`orchestrate_s2v1.sh`,
`verify_and_report_s2v1.py`) are in git history at commit `f74e5ed`.

Last completed session: `s2_blocks_v1` (2026-07-26, $4.49, unattended end to
end) — `logs/experiments/2026-07-26_stage3_s2_blocks_v1_gpu_run.md`.
Current session target: the **start-point ablation**, two arms on one pod
(`logs/proposals/2026-07-27_stage3_start_point_ablation.md`).

## Files

| file | role |
| --- | --- |
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
   "stuck" (`logs/experiments/2026-07-26_runpod_pod_readiness_misdiagnosis.md`).

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
