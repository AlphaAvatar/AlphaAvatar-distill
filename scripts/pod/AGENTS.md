# scripts/pod — GPU pod session scripts (s2_blocks_v1)

Prepared 2026-07-26 for the approved `s2_blocks_v1` recovery run; the
session was **paused before execution** because RunPod containers refused
to start (see `logs/STATE.md`). Everything here is ready to run unchanged.

## Resume procedure

1. `runpodctl pod create --template-id runpod-torch-v240 --gpu-id
   "NVIDIA L40S" --container-disk-in-gb 40 --volume-in-gb 120
   --name aadistill-s2v1 --terminate-after <utc-now+8h>`
   (RTX 6000 Ada Generation, 48 GB, $0.84/h, is a verified-equivalent
   fallback; any 40 GB+ Ampere-or-newer card works — the venv brings its
   own cu128 torch, the container image is irrelevant.)
2. Wait until `uptimeSeconds > 0` and `runpodctl ssh info <id>` answers.
   **2026-07-26 failure mode:** three pods sat at `uptimeSeconds: 0` for
   15+ min (host never started the container, billing runs anyway). If
   that happens, delete after ~8 min and re-allocate (different pool/GPU).
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
