# External artifact manifests

Manifests for artifacts stored outside GitHub (AGENTS.md 2.5). All entries
in `AlphaAvatar/aadistill-artifacts` are **private** storage/transfer
artifacts, not public releases.

## stage3/s2_blocks_v1 — mixture-v1 recovery final checkpoint (current best)

- **Artifact:** same repo, path `stage3/s2_blocks_v1/`: `step_002700/model/`
  (final fp32 checkpoint, 2.3 GB) + `train_log.jsonl`, `run_manifest.json`,
  `eval_holdout_v1.json`, `eval_holdout_v1_int8.json`,
  `eval_holdout_v1_int8_decoder.json`, `gen_smoke.json`, `console.log`,
  `s2v1_artifact_hashes_2026-07-26.txt` (14 files).
- **Revision:** `b1b5170cb45ce7b141c02c23ca4b1bb89918a85b` (2026-07-26 15:45 UTC,
  last of the run's upload commits).
- **Hashes (sha256):** `step_002700/model/model.safetensors`
  `f275fbfd13b43b61629f92b67a7b9586d34deeea977b6c8e197c708ecd4f591f`;
  full per-file list in `artifacts/stage3/s2v1_artifact_hashes_2026-07-26.txt`
  (local, gitignored; computed pod-side before upload) and mirrored in the repo
  at `stage3/s2_blocks_v1/s2v1_artifact_hashes_2026-07-26.txt`. Upload verified
  independently after the fact — LFS sha256 for the large weights, download+hash
  for small files, **14/14 match** (`scripts/pod/verify_and_report_s2v1.py
  verify`). Weights are HF-only; small files also live locally under
  `artifacts/stage3/s2_blocks_v1/`.
- **License:** internal artifact, same derivation as the entries below, plus the
  `stage2_offline_v1` mixture (permissive public sources; synthetic provenance
  and share-alike notes recorded in `data/stage2_v1/stage2_offline_v1.manifest.json`).
  Not for redistribution as-is (no model card).
- **Creation command:** `uv run python scripts/train_stage3.py --config
  configs/stage3_s2_blocks_v1.json` (run `s2_blocks_v1`, config sha256
  `5a61689cb9a8…`, code state git `f73be5516a85` + logged uncommitted diff
  `2e04f6834922…`, pod `ippwmpc8wzed24`, 1× L40S, 2026-07-26); uploaded by
  `scripts/pod/post_run.sh` in the same session.
- **Related logs:** `experiments/2026-07-26_stage3_s2_blocks_v1_gpu_run.md`.
  Holdout_v1 NLL 3.8003 (bf16) — **current best student checkpoint**; it is also
  the reference arm (`A0 chain`) of the proposed start-point ablation.
- **Not retained:** optimizer state and rolling checkpoints (deleted with the pod).

## stage3/s1_ffn_norm_v0/step_000660 — s1 recovery final checkpoint

- **Artifact:** `https://huggingface.co/AlphaAvatar/aadistill-artifacts`
  (private, model repo), path `stage3/s1_ffn_norm_v0/step_000660/model/`
- **Revision:** commit `727c837e810ef58eefd5e5553155b459f21414e5` (2026-07-25)
- **Size:** 2.3 GB (fp32 safetensors) + tokenizer/config files
- **Hashes (sha256):** `model.safetensors`
  `dc64f244e203d607ed2ea63836400287878b26b96ebf42dc19ec70f154ace900`
  (bit-identical to the pod-produced original; full per-file list verified
  before upload and re-verified after pod download on 2026-07-25)
- **License:** internal artifact derived from teacher
  `Qwen/Qwen3-4B-Thinking-2507` (Apache-2.0) and the permissive-source
  `stage2_offline_v0` mixture; not for redistribution as-is (no model card).
- **Creation command:** `uv run python scripts/train_stage3.py --config
  configs/stage3_s1_ffn_norm.json` (run `s1_ffn_norm_v0`, commit `96c30ce`);
  uploaded 2026-07-25 with `hf upload AlphaAvatar/aadistill-artifacts
  artifacts/stage3/s1_ffn_norm_v0/checkpoints/step_000660/model
  stage3/s1_ffn_norm_v0/step_000660/model` (user-approved 2026-07-25).
- **Related logs:** `experiments/2026-07-22_stage3_s1_gpu_run.md`;
  purpose: durable external storage + transfer vehicle for the sub-stage 2
  A/B session (decision record 2026-07-25).

## stage3/s1_ext_v0 and stage3/s2_blocks_v0 — sub-stage 2 A/B finals

- **Artifact:** same repo, paths `stage3/s1_ext_v0/` and
  `stage3/s2_blocks_v0/`: `step_000660/model/` (final fp32 checkpoint,
  2.3 GB each) + `train_log.jsonl`, `run_manifest.json`,
  `eval_holdout_v1.json`, `gen_smoke.json`, `console.log` per arm.
- **Revision:** `526caa780132dfcc522fcd1f8093fa7351e0db0c` (2026-07-25).
- **Hashes (sha256):** full per-file list in
  `artifacts/stage3/ab_artifact_hashes_2026-07-25.txt` (local, gitignored;
  computed pod-side before upload). These weights are HF-only (no local
  copy); small files verified locally after direct download.
- **License:** internal artifacts, same derivation as the s1 entry above.
- **Creation command:** `uv run python scripts/train_stage3.py --config
  configs/stage3_s1_ext.json` / `--config configs/stage3_s2_blocks.json`
  (commit `6230a14`, pod `simbeepnf8syuu`, 2026-07-25).
- **Related logs:** `experiments/2026-07-25_stage3_s2_ab_gpu_run.md`.
