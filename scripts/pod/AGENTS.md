# scripts/pod — GPU pod session scripts (s2_blocks_v1)

Prepared 2026-07-26 for the approved `s2_blocks_v1` recovery run. The first
attempt was paused because pods appeared not to start; that was a
**misdiagnosis of two broken readiness signals** — the pods were healthy.
Root cause and evidence:
`logs/experiments/2026-07-26_runpod_pod_readiness_misdiagnosis.md`.
Everything here is ready to run unchanged.

## Resume procedure

1. `runpodctl pod create --image
   runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 --gpu-id "NVIDIA L40S"
   --container-disk-in-gb 40 --volume-in-gb 120 --ports "22/tcp,8888/http"
   --name aadistill-s2v1 --terminate-after <utc-now+8h>`

   **`--ports "22/tcp"` is mandatory** (see step 2). Prefer **L40S** for
   this project: s1 (2026-07-22) and the A/B (2026-07-25) both ran on L40S,
   so keeping it holds throughput/memory numbers comparable. RTX 6000 Ada
   (48 GB, $0.84/h) is a verified working fallback if L40S is unavailable —
   any 40 GB+ Ampere-or-newer card works, since the venv brings its own
   cu128 torch and the container image is irrelevant — but note the
   hardware change in the experiment log if you use it.

2. Wait for readiness. **Do NOT trust `runpodctl pod get`'s
   `uptimeSeconds`: in runpodctl 2.7.1 it is always `0`, even on a healthy
   running pod.** And **`runpodctl ssh info` returning `"pod not ready"`
   does not mean the pod failed** — it means no public TCP 22 mapping
   exists, which never appears if the pod was created without
   `--ports "22/tcp"`. Both were verified against the API on 2026-07-26;
   together they cost ~$0.95 in healthy pods deleted as "stuck".

   Use GraphQL, and treat an actual SSH connection as the only ground truth:

   ```bash
   KEY=$(python3 -c "import re;t=open('$HOME/.runpod/config.toml').read();\
   print(re.search(r'apikey\s*=\s*(.+)',t,re.I).group(1).strip().strip('\"').strip(\"'\"))")
   curl -s -X POST "https://api.runpod.io/graphql?api_key=$KEY" \
     -H 'Content-Type: application/json' \
     -d '{"query":"query { pod(input:{podId:\"<id>\"}) { runtime { uptimeInSeconds ports { ip publicPort privatePort type } } } }"}'
   ```

   `runtime: null` = still starting; non-null `uptimeInSeconds` = container
   up. Poll until a `tcp`/`privatePort 22` entry appears and use that
   `ip:publicPort`. Config values are **single-quoted** and the key is
   lowercase `apikey` — strip `'` as well as `"`, or GraphQL just returns
   `{"error":{}}`.

   If a pod was already created without ports, recover it in place instead
   of re-allocating: `runpodctl pod update <id> --ports "22/tcp,8888/http"`
   then `runpodctl pod restart <id>` (mapping appears ~30 s later).
3. `scp` (port from ssh info): `setup.sh`, `train.sh`, `post_run.sh`,
   `hashes_transfer.txt`, `hashes_ckpt.txt` → `/workspace/`, plus write
   the HF token (from `hf auth token` on the dev box) to
   `/workspace/hf/token` (mode 600). Never echo the token into logs.
4. `bash /workspace/setup.sh` → follow `MARKER:` lines in
   `/workspace/setup.log` (`ENV_READY` → `CKPT_READY` → `TESTS_PASSED` →
   `SETUP_DONE`; any `SETUP_FAILED:<step>` fails loudly).
5. `bash /workspace/train.sh` → training detached; markers appended to
   `/workspace/run_markers.log` (`TRAIN_DONE` / `TRAIN_FAILED`), console
   at `/workspace/console_s2v1.log`. ~2700 steps ≈ 2.3 h on L40S-class.
6. `bash /workspace/post_run.sh` → gate evals (bf16 holdout on GPU, INT8
   fake-quant on CPU, generation smoke) + hash + upload to
   `AlphaAvatar/aadistill-artifacts` under `stage3/s2_blocks_v1/`
   (`MARKER:POST_DONE`).
7. From the dev box: verify the HF listing (file names + sizes) before
   `runpodctl pod delete`.

## Inputs already staged (private HF repo `AlphaAvatar/aadistill-artifacts`)

- `transfer/repo_20260726.bundle` — git bundle @ commit `f73be55` (main).
  sha256 in `hashes_transfer.txt`.
- `transfer/stage2_data_20260726.tar.zst` — data/stage2_v1 + data/stage2
  splits + holdout_v1.jsonl (26.5 MB). sha256 in `hashes_transfer.txt`.
- `stage3/s2_blocks_v0/step_000660/model/` @ revision `526caa78…db0c` —
  start checkpoint; per-file sha256 in `hashes_ckpt.txt` (from
  `artifacts/stage3/ab_artifact_hashes_2026-07-25.txt`).

If the repo has moved past `f73be55` when resuming, regenerate the bundle
(`git bundle create … HEAD main`), re-upload, and update
`hashes_transfer.txt` — setup.sh verifies hashes and will fail loudly on
a stale bundle.
