# External artifact manifests

Manifests for artifacts stored outside GitHub (AGENTS.md 2.5). All entries
in `AlphaAvatar/aadistill-artifacts` are **private** storage/transfer
artifacts, not public releases.

Two reading notes:

* **Creation commands are recorded as run.** Script and config paths changed in
  the 2026-07-30 repository reorganization (`633dc6b`): `scripts/train_stage3.py`
  → `scripts/training/train_stage3.py`, `configs/stage3_*.json` →
  `configs/stage3/*.json`, and so on. See
  [`docs/REPO_LAYOUT.md`](../docs/REPO_LAYOUT.md) for the current map.
* **Per-run experiment logs were consolidated** into
  [`EXPERIMENTS.md`](EXPERIMENTS.md) on 2026-07-31 (`1fbcb99`); the originals are
  in git history at `866dac2`. "Related logs" below point at the consolidated
  record.

## e5_start/ — the Experiment 5 corpora (arm C and arm R)

- **Artifact:** `AlphaAvatar/aadistill-artifacts`, prefix `e5_start/`.
  - `e5_arm_c.tar.gz` — 4.4 MB, sha256
    `fdf44b34a89164a88c59256129e692fd89021bcf53209831ca6d8d9eb6e49bee`.
    Both seeds' arm-C corpora: 2,294 examples each, 1,034 blocks, 905,488
    candidate CE tokens, 362 system blocks. Per-file:
    `sa/examples.jsonl` `4bc23c3f…`, `sb/examples.jsonl` `03c9ba9a…`,
    `system_ids.json` (identical for both seeds) `18ace28a…`.
  - `e5_arm_r.tar.gz` — 4.5 MB, sha256
    `e2cbbd45eefa98b142911414036e6b77fa926024dea368e2e7bc2d075b2c8e96`.
    Both seeds' arm-R corpora from attempt 5: **2,098** (sa) and **2,042** (sb)
    records. Per-file: `sa/examples.jsonl` `247941ab…`, `sa/manifest.json`
    `eea4f1a3…`, `sa/system_ids.json` `f89a3386…`; `sb/examples.jsonl`
    `ae1a4203…`, `sb/manifest.json` `9efdddcd…`, `sb/system_ids.json`
    `6ab61945…`.
  - `e5_arm_r_corpora.tar.gz` — 4.3 MB, sha256 prefix `867029bece693a6c`, pushed
    by the pod itself at `CORPORA_RETAINED` on attempt 8. Same content lineage,
    kept as the in-run copy.
- **Created:** arm R generated on attempt 5 (2026-08-07, ~$1.24 of the $1.55) from
  `Qwen/Qwen3-4B-Thinking-2507@768f209d…` over student rollouts from
  `p2_ceheavy_{sa,sb}`, preset `{temperature 0.6, top_p 0.95, top_k 20, min_p 0}`,
  sessions corpus `2b4edc2e…`. Arm C is a mask move over the same teacher
  trajectories and needs no generation.
- **Verification:** `scripts/data/verify_staged_r.py` asserts all of the above on
  the pod before anything trains — record counts, every file hash, teacher id and
  revision, the P2 checkpoint identity by *weight* hash, tokenizer and
  chat-template hashes, decoding preset, sessions hash, deterministic sha256 seed
  derivation, and that every record reloads through `example_to_rendered`. All 36
  checks passed on attempts 6, 7 and 8. Every check is fatal.
- **Why it is kept:** these corpora cost GPU time twice before surviving — lost at
  teardown on attempt 1 (records without token payloads) and again on attempt 4
  (the side bundle shipped manifests only). Reuse is artifact reuse, not reuse of
  an experimental outcome: attempt 5 produced no training result.

## E5 trained checkpoints — LOST

- **Artifact:** none. `e5_{c,r}_{sa,sb}` at `step_001356` were trained on attempt
  8 (117 minutes of L40S) and never left the pod: the launcher fetched
  `step_000738`, a constant from the superseded 492-block design, so the copy
  matched nothing and the weights died with the pod.
- **Consequence:** the E5 result stands on the retained evaluation artifacts
  (reports, per-sample generations, feasibility report, gate records). Any
  *re-evaluation* of C or R would require retraining, at roughly $2.90.
- **Fixed:** the step tag is now derived from the feasibility report's measured
  `optimizer_steps`, guarded by a test, and the remote `find` was moved out of the
  single quotes where the variable could never have expanded.

## e1_scaling_20260801 — Experiment 1 checkpoints and evaluation results

- **Artifact:** same repo, prefix `e1_scaling_20260801/`. Per arm:
  `step_*/model/` (fp32 checkpoint), `train_log.jsonl`, `run_manifest.json`,
  `eval_holdout_v1.json`, `gen_smoke.json`, `console.log` and a pod-side sha256
  list. Evaluation results for all 25 checkpoints are under
  `e1_scaling_20260801/_evaluation/` (75 JSON summaries: behaviour, GSM8K,
  holdout NLL).
- **Created:** 2026-08-01/03 across four L40S pods. Training $47.6, control +
  first evaluation $8.1, full sweep $5.8.
- **Coverage:** **20 of 25 checkpoints hold weights on the relay.** Five do not —
  `e1_r2960k_sb_pca`, `e1_r5500k_sb_pca`, `e1_r2960k_sb_rand`,
  `e1_r5500k_sb_rand` and the step-matched control
  `e1_ctl_r0250k_sa_pca_stepmatched` — because the relay hit its private LFS
  storage limit mid-session and the squash was never credited (see the 2026-08-02
  decision). All five are held on the dev box under
  `artifacts/stage3/rescued/<arm>/`, each **hash-verified 6/6 against its
  pod-side manifest**, with the manifest retained alongside as `pod_hashes.txt`.
  The small evaluation JSONs uploaded successfully because they are not LFS
  objects.
- **Evaluation protocol pinned with the results:** uncapped generation within an
  effective context of **8,192** derived from the trained `block_len` (the
  derivation is recorded in every summary and every per-sample record — this is
  *not* a 262K-context evaluation), greedy decoding, mandatory system message
  with a sample's own preserved when present, chat template sha256 `3802169b…`
  identical to corpus-generation time, and a fixed degeneration detector applied
  with the same thresholds to every checkpoint.
- **License/provenance:** internal artifacts derived from the Apache-2.0 teacher
  and the permissive-source corpus v2. Not for redistribution as-is.
- **Related logs:** [`EXPERIMENTS.md`](EXPERIMENTS.md) §11,
  [`STATE.md`](STATE.md) §12–13, `artifacts/stage3/e1_consolidated.json`.

## recovery corpus v2 + token ladders

- **Artifact:** the Stage 3 recovery corpus and its six-rung nested token
  ladders, built 2026-08-01: `sessions.jsonl` (74 MB, 11,174 accepted sessions),
  `candidates.jsonl` (857 MB, all 4 candidates per example with per-candidate
  verdicts), `manifest.json`, plus **two packs** — the capability-gap weighted
  cut (`blocks.npz` 14 MB, `audit.jsonl` 6 MB, `ladder.json`; 3,720 blocks) and
  the **uniform cut used by Experiment 1**, re-cut the same day at
  `artifacts/stage3/ladder_uniform_probe` (3,715 blocks; `blocks.npz` sha256
  `6f324cb0f37bc0f07128e554ce8c161879419537478950496534f75fcecb249c`,
  `ladder.json` `d4941722a099754ef5ba82d529c0fe2274a6b97af61b2fa52519ed22b84201f0`,
  `audit.jsonl` `15f16b7b22b229e9c0ae510b85b4c967948828e0e7d50fac15c3e516e3e911e6`).
  Both packs derive from the same `sessions.jsonl` and are regenerable from it
  in ~15 min of CPU.
- **Storage: on the relay under `stage3_recovery_corpus_v2/`** (9/9 files
  hash-verified 2026-08-01, [`STATE.md`](STATE.md) §2) **and** on the dev box at
  `/tmp/claude-1000/-home-ecs-user-AlphaAvatar-distill/2e9e81e1-…/scratchpad/bulk/`.
  This entry previously read "NOT YET PERSISTED", contradicting `STATE.md`;
  corrected 2026-08-03 after both `candidates.jsonl` (`f7f5035e…`) and
  `sessions.jsonl` (`2b4edc2e…`) were re-hashed against the local copies and
  matched.
- **Hashes (sha256):** `candidates.jsonl`
  `f7f5035ef8b42fb4bacd4f28692d214ff734f4ff820c1f6e37436e328546ecc7`;
  `sessions.jsonl`
  `2b4edc2e2cc16cd56dae3d340345e1a17e2c4a8baa9837650a7bf5e340fa6fcd`.
  Both recorded in `manifest.json` by the builder and **re-verified against the
  local files on 2026-08-01** — match.
- **Teacher:** `Qwen/Qwen3-4B-Thinking-2507@768f209d9ea81521153ed38c47d515654e938aea`,
  bf16, vLLM 0.26.0, torch 2.11.0+cu130, transformers 5.14.1, 1× L40S.
  Tokenizer sha256 `3ec3c124…`, chat template sha256 `3802169b…`, stop ids
  `[151643, 151645]` from `generation_config.json`, preset
  `0.6 / 0.95 / top_k 20 / min_p 0`, `n=4`, seed base `20260731`
  (`seed + batch_index + candidate_index × 1000003`).
- **Creation command:** `scripts/rollout/build_recovery_corpus.py --engine vllm
  --n 4 --block-len 8192 --batch-size 256 --select stride --limits
  gsm8k=1700,openmath=900,code=1200,tool_calling=2600,rag_evidence=4100,multihop_qa=1074
  --max-hours 26 --out /workspace/out/bulk`, then
  `scripts/data/build_token_ladder.py --sessions … --block-len 8192 --mixture
  gsm8k=0.22,openmath=0.17,code=0.16,tool_calling=0.15,rag_evidence=0.20,multihop_qa=0.10
  --out /workspace/out/packed`, then `scripts/data/validate_corpus_gate.py`
  (gate: PASS).
- **Known gap (P4):** the manifest's `code_state` block carries **no git
  commit** — the bundle was unpacked outside a git checkout and `git rev-parse`
  failed, so `code_state_error` was stored instead. Code state is pinned only by
  the shipped bundle.
- **License/provenance:** derived from `data/stage2_v1` prompts (public
  datasets, licenses recorded in the Stage 2 v1 manifest) plus generations from
  the Apache-2.0 teacher. No user data, no secrets. Not reproducible
  token-for-token (sampled decoding; bf16 decoding is not batch-invariant) — the
  hashes pin the experiment (P5), not a re-derivable procedure.
- **Related logs:** [`EXPERIMENTS.md`](EXPERIMENTS.md) §10, [`STATE.md`](STATE.md)
  §6–§8, and the three 2026-08-01 [decision records](decisions.md).

## stage3_teacher_corpus_20260730 — teacher corpus v1 (superseded)

- **Artifact:** same repo, prefix `stage3_teacher_corpus_20260730/`:
  `candidates.jsonl`, `targets.jsonl`, and `rollout_snapshot/`
  (`rollouts.jsonl` + `manifest.json`, 1,504 rollouts / 2.46M tokens).
- **Created:** 2026-07-30. 752 prompts → **540 accepted** targets; targets
  sha256 `18028f0c…`, rollout snapshot sha256 `0e5b20dd…`.
- **Status: superseded by corpus v2.** Sampled at temperature 1.0 / top_p 1.0 /
  top_k off (not the official preset) and **effectively n=1** — 92.7% of
  candidate pairs are byte-identical because the serving engine seeded per
  request, not per candidate. Retained as labelled auxiliary data only
  ([`PROPOSAL.md`](archive/PROPOSAL.md) §2); the 4,096-token cap censored 19.9% of
  rollouts (69.7% of `openmath`).
- **License/provenance:** as the corpus v2 entry above.
- **Related logs:** [`EXPERIMENTS.md`](EXPERIMENTS.md) §5, §6.

## tt2x2/ and ttb/ — the two 2026-07-30 four-arm runs (diagnostics)

- **Artifact:** same repo, prefixes `tt2x2/` (post-s2v1 continuation, 4 arms)
  and `ttb/` (corrected baseline from the Stage 1 init, 4 arms). Per arm:
  final checkpoint, `train_log.jsonl`, `run_manifest.json`, eval JSONs,
  generations, `console.log`, and a pod-side sha256 list
  (`*_artifact_hashes_20260730.txt`).
- **Created:** 2026-07-30, 1× L40S. `tt2x2` $3.50, `ttb` $2.30.
- **Status: diagnostics, not route decisions.** `tt2x2` forked every arm from a
  public-trained checkpoint (invalid as a target comparison); `ttb` forked
  correctly from the Stage 1 init but was convergence-limited (137 steps) and
  measurement-limited (99.3% of treatment generations censored at 512 tokens).
  Neither supports a claim about teacher-native supervision.
- **Hashes:** per-arm sha256 lists are committed alongside each arm on the relay.
- **Related logs:** [`EXPERIMENTS.md`](EXPERIMENTS.md) §5.

## stage3/s2v1_from_init and stage3/s2v1_from_s1 — start-point ablation finals

- **Artifact:** same repo, paths `stage3/s2v1_from_init/` and
  `stage3/s2v1_from_s1/`: `step_002700/model/` (final fp32 checkpoint, 2.3 GB
  each) + `train_log.jsonl`, `run_manifest.json`, three holdout eval JSONs,
  `eval_behavior_v0.json`, `eval_behavior_v0.generations.jsonl`,
  `gen_smoke.json`, `console.log`, and the pod-side sha256 list (16 files each).
  Reference behavior scorecards for `s1_ffn_norm_v0@660` and
  `s2_blocks_v1@2700`, scored on the same GPU in the same session, are under
  `stage3/reference_scorecards/`.
- **Revision:** uploaded 2026-07-27 by `scripts/pod/post_run.sh`; upload
  independently verified after the fact — **16/16 files match pod sha256 for
  each arm** (`scripts/pod/verify_and_report.py verify --run <arm>`).
- **Hashes (sha256):** per-file lists committed at
  `artifacts/stage3/s2v1_from_init_artifact_hashes_20260727.txt` and
  `artifacts/stage3/s2v1_from_s1_artifact_hashes_20260727.txt` (local,
  gitignored; computed pod-side before upload) and mirrored in the repo under
  each arm's prefix.
- **License:** internal artifacts, same derivation as the entries below
  (teacher Apache-2.0 + the permissive-source `stage2_offline_v1` mixture).
  Not for redistribution as-is (no model card).
- **Creation command:** `uv run python scripts/train_stage3.py --config
  configs/stage3_s2v1_from_init.json` (config sha256 `b2520a2e0ad8…`) and
  `--config configs/stage3_s2v1_from_s1.json` (`7e0612ccf3aa…`), code state git
  `f3d7547`, pod `ruib84xvfyieqm`, 1× L40S, 2026-07-27.
- **Related logs:** [`EXPERIMENTS.md`](EXPERIMENTS.md) §3 (start-point ablation).
  `s2v1_from_init` holdout_v1 NLL **3.8285** — **the recommended branch point**
  for further recovery work (best behavior scorecard, 2700 total steps);
  `s2v1_from_s1` **3.8067**.
- **Not retained:** optimizer state and rolling checkpoints (deleted with the pod).

## stage1/qwen3_0p6b_init_v0 — Stage 1 init checkpoint (start point of arm A2)

- **Artifact:** same repo, path `stage1/qwen3_0p6b_init_v0/checkpoint/`
  (fp32 safetensors, 1.2 GB + tokenizer/config files, 6 files).
- **Revision:** `b955bd2f79b03a5418e2b8ca518a35faf047f085` (2026-07-27).
- **Hashes (sha256):** `model.safetensors`
  `86fbba78e8a2a32481ca77e5ac362ed1f17a39dbc30bcbc952cabd5df2633e54`;
  full per-file list in `scripts/pod/hashes_ckpt.txt` (committed), verified
  against the local copy before upload and re-verified pod-side by `setup.sh`.
- **License:** internal artifact derived from teacher `Qwen/Qwen3-4B-Thinking-2507`
  (Apache-2.0) via the Stage 1 initialization recipe; not for redistribution
  as-is (no model card).
- **Creation command:** `uv run python scripts/init_stage1.py --config
  configs/stage1_qwen3_0p6b_from_4b_thinking.json` (2026-07-14); uploaded
  2026-07-27 with `hf upload AlphaAvatar/aadistill-artifacts
  artifacts/stage1/qwen3_0p6b_init_v0/checkpoint
  stage1/qwen3_0p6b_init_v0/checkpoint`.
- **Related logs:** [`EXPERIMENTS.md`](EXPERIMENTS.md) §2 (Stage 1);
  purpose: transfer vehicle for the `from_init` arm of the start-point ablation
  (consolidated into [`EXPERIMENTS.md`](EXPERIMENTS.md) §3). It was the only
  start point not previously on the relay.

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
  verify` — renamed 2026-07-27 to `scripts/pod/verify_and_report.py`;
  pre-rename file in git history at `f74e5ed`). Weights are HF-only; small
  files also live locally under `artifacts/stage3/s2_blocks_v1/`.
- **License:** internal artifact, same derivation as the entries below, plus the
  `stage2_offline_v1` mixture (permissive public sources; synthetic provenance
  and share-alike notes recorded in `data/stage2_v1/stage2_offline_v1.manifest.json`).
  Not for redistribution as-is (no model card).
- **Creation command:** `uv run python scripts/train_stage3.py --config
  configs/stage3_s2_blocks_v1.json` (run `s2_blocks_v1`, config sha256
  `5a61689cb9a8…`, code state git `f73be5516a85` + logged uncommitted diff
  `2e04f6834922…`, pod `ippwmpc8wzed24`, 1× L40S, 2026-07-26); uploaded by
  `scripts/pod/post_run.sh` in the same session.
- **Related logs:** [`EXPERIMENTS.md`](EXPERIMENTS.md) §3 (mixture v1).
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
- **Related logs:** [`EXPERIMENTS.md`](EXPERIMENTS.md) §3 (s1 FFN+norm);
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
- **Related logs:** [`EXPERIMENTS.md`](EXPERIMENTS.md) §3 (sub-stage 2 A/B).

## engine_bench_20260729 — engine benchmark + teacher-corpus pilot

- **Artifact store:** `AlphaAvatar/aadistill-artifacts` (private), prefix `engine_bench_20260729/`
- **Created:** 2026-07-29, pod `g8ajahpwirhrfx` (1× L40S, $0.99/h, ~2.6 h, ≈$2.60)
- **Creation commands:**
  - `python scripts/bench_engines.py --engines hf --n-prompts 10 --max-new-tokens 4096 --hourly-usd 0.99 --out artifacts/bench/engines_v0`
  - `python scripts/generate_teacher_answers.py --engine hf --limit-per-slice 10 --n 4 --batch-size 4 --max-new-tokens 4096 --max-hours 2.2 --out artifacts/stage2_v2/pilot`
- **Teacher:** `Qwen/Qwen3-4B-Thinking-2507@768f209d`, bf16
- **Experiment log:** [`EXPERIMENTS.md`](EXPERIMENTS.md) §4 (engine benchmark)

| file | size | sha256 |
| --- | --- | --- |
| `bench/report.json` | 3.3 KB | (see `code_state` block inside) |
| `bench/decision.json` | 202 B | winner `hf`, rule R2 |
| `pilot/candidates.jsonl` | 1.38 MB | `169cece8d02bbad469aa435161942af1f9316012276de03aa80422cc5b6bd821` |
| `pilot/targets.jsonl` | 250 KB | `2a5986490f8b5ee8eb25eccbc8c49dabd33e418d0606b1b4d1650d514a06573e` |
| `pilot/manifest.json` | 2.9 KB | carries both hashes above, `complete: true`, 50/50 prompts |

**License/provenance:** derived from `data/stage2_v1` prompts (public datasets,
licenses recorded in the Stage 2 v1 manifest) plus generations from a public
Apache-2.0 teacher. No user data, no secrets.

**Caveat:** the corpus is **not** reproducible token-for-token. Sampling is
untruncated at temperature 1.0, and this session measured that even *greedy*
bf16 decoding is not batch-invariant on this teacher — so the hashes above pin
the experiment (P5), not a re-derivable procedure.

## corpus_v2_clean + rung_0860k_clean_median — Experiment 2 phase 1 (D1)

- **Artifact:** `artifacts/stage3/corpus_v2_clean/` (`sessions_clean.jsonl` sha256
  `9cbdd59acc44f64c44623a2351b7d2dc799097e513d40cd4eff4f2e9b0d67bd6`, 68 MB;
  `cleaning_audit.json`
  `c70c85288df7a0ca487965321b6c7a2bb33eefee3b112a9ed337e923d2af543d`;
  `cleaning_per_example.jsonl`; `d0_session_order.txt`) and the packed rung
  `artifacts/stage3/rung_0860k_clean_median/` (`blocks.npz`
  `d87a4688f723c97ce7a6d6fcb80ac9d93ef83236ede205d10041c3b9aeb9003c`,
  `ladder.json`
  `8dc700095abf5560dbb9f0ccea47a9bb00992dfcd3f03df85ac0457c70022a1b`,
  `audit.jsonl`
  `fb10f9faf6f232109779dc23033c82c4ee5db7f6ec7e9f0e9035ab00fcc7a38e`, 2.7 MB
  total). Comparison records: `artifacts/stage3/e2_d1_corpus_audit.json`
  `04d04378269e4881c19265a45925670bc31edebd7f118da9bfbff851bf9494ee` and
  `artifacts/stage3/e2_selection_rule_audit.json`
  `70c380d469400ffe518d9bed9c92b1c6d8068e2263ab0bff69168b26b6c6b58a`.
- **Created:** 2026-08-03, **CPU only, $0**. ~4 min to screen 11,174 sessions,
  ~10 min to pack. No teacher inference: every target is a completion already
  present in corpus v2's retained `n=4` candidates, and KD teacher distributions
  are computed online at training time, so a changed target needs no logit
  recomputation.
- **Derivation:** `scripts/data/build_cleaned_corpus.py --selection median`
  (rules `aadistill.data.cleaning`, `RULES_VERSION = "clean-v2"`) over
  `candidates.jsonl` `f7f5035e…` + `sessions.jsonl` `2b4edc2e…`, then
  `scripts/data/build_matched_rung.py --control-rung 860000` against
  `artifacts/stage3/ladder_uniform_probe`, uniform mixture, block-len 8192,
  pool overshoot 1.20.
- **Contents:** 10,778 of 11,174 sessions cleaned (96.5%). The rung is **682
  blocks / 1,023 optimizer steps / 5,586,944 packed tokens — identical to
  Experiment 1's 0.86M PCA control** — at 858,409 supervised tokens (−0.733%),
  1,479 sessions, **89.1% prompt overlap** with that control, per-type share
  drift ≤ 0.17 pp. The control's 16 validation blocks are appended verbatim
  (token-ids sha256 `4d36705cfcf414af…`, 81,195 supervised tokens) and verified
  byte-identical through `aadistill.data.ladder`.
- **A second corpus exists for comparison only:**
  `artifacts/stage3/corpus_v2_clean_shortest/`, identical gates with
  `--selection shortest`. Never trained on; it exists so the two selection rules
  could be measured against each other rather than argued about.
- **Tokenizer/template:** teacher `Qwen/Qwen3-4B-Thinking-2507@768f209d`, vocab
  sha256 `3ec3c124…`, chat template `3802169b…` — both reproduced exactly on the
  dev box under transformers 5.13.1 against the corpus's 5.14.1.
- **Status: prepared, NOT trained on.** Awaiting launch approval
  ([`PROPOSAL.md`](archive/PROPOSAL.md)).
- **License/provenance:** derived from corpus v2; same constraints. No new
  generation, no new sources.
- **Related logs:** [`PROPOSAL.md`](archive/PROPOSAL.md) §3–§4,
  [`EXPERIMENTS.md`](EXPERIMENTS.md) §12.

## e1_r0860k_s{a,b}_pca run records — recovered from the relay

- **Artifact:** `artifacts/stage3/rescued/_relay/e1_r0860k_s{a,b}_pca/` —
  `run_manifest.json`, `train_log.jsonl`, `eval_holdout_v1.json`,
  `gen_smoke.json` and the pod-side hash list, fetched 2026-08-03 from
  `AlphaAvatar/aadistill-artifacts` under `e1_scaling_20260801/`.
- **Why:** these are the Experiment 2 control's authoritative records. The
  dev-box copies were destroyed by the Experiment 1 `scp` basename collapse; the
  relay kept per-arm copies. They restore the full 10-point val-CE trajectories
  and let the committed configs be verified against the run manifests
  (`config_sha256` `08264ef1…` and `9048173d…`, both reproduced).
- **Also on the relay, unchanged:** `step_001023/model/` for both arms — the D0
  weights Experiment 2 will re-evaluate on any new capability set.
- **Related logs:** [`PROPOSAL.md`](archive/PROPOSAL.md) §2,
  [`EXPERIMENTS.md`](EXPERIMENTS.md) §12.2.

## battery_v2 — the frozen Experiment 2 capability battery

- **Artifact:** `artifacts/eval/battery_v2/` — seven jsonl sets plus
  `manifest.json` (sha256
  `060bdd3170c5cfe0cdb749a7bf32e6d264d943085f0d24717f8b86d5706561df`).
  Per-set sha256: `knowledge` `2d4420ce…` (150) · `math_verified`
  `bf73cb4a…` (100) · `gsm8k` `1ad4ad22…` (100) ·
  `multihop` `3bd25d89…` (100) · `rag` `1e31c9e0…` (100) ·
  `answerability_paired` `1a436932…` (120 = 60 pairs) ·
  `safety_paired` `ee73e208…` (100 = 50 pairs). Plus the
  76-prompt `data/eval_behavior_v0/prompts.jsonl` reused verbatim. **846 total.**
- **Created:** 2026-08-03, **CPU only, $0**, by
  `scripts/data/build_capability_battery.py`.
- **Supersedes `battery_v1`** (746 prompts, `capability-v1`), which used SQuAD-v2
  pairs as its refusal set. That measured evidence-conditioned answerability on
  benign prompts, not safety refusal; the set is renamed `answerability_paired`
  and a distinct XSTest-based `safety_paired` was added. `battery_v1` was deleted
  rather than retained — it was never used to score any model output.
- **Sources and revisions:** `mandarjoshi/trivia_qa` `rc.nocontext` validation
  (Apache-2.0) · `HuggingFaceH4/MATH-500` test (MIT) · `openai/gsm8k` `main` test
  (MIT) · `hotpotqa/hotpot_qa` `distractor` validation (CC-BY-SA-4.0) ·
  `rajpurkar/squad_v2` validation (CC-BY-SA-4.0) · `Paul/XSTest` train
  (CC-BY-4.0). Exact sample ids are in the manifest.
- **Scorers:** `src/aadistill/evaluation/capability.py`,
  `src/aadistill/evaluation/strict_answer.py` and the degeneration detector, all
  hashed in the manifest. Deterministic only — no LLM judge is a primary scorer.
  Both paired sets are reported and gated on **pair accuracy**.
- **Leakage:** 0 collisions, checked structurally and by the corpus's own
  `content_key`/`prompt_key` rule against 65,913 content, 59,113 reserved-prompt
  and 10,128 corpus-v2-prompt hashes. A self-test confirms a real corpus-v2
  prompt does hash into the exclusion set. **Item-level exclusion only** —
  `knowledge`, `math_verified` and `safety_paired` are source-disjoint; `gsm8k`
  is split-held-out; `multihop`, `rag` and `answerability_paired` are
  split-held-out, near-domain item-disjoint. No out-of-domain claim is made.
- **Validation:** 112 CPU tests, including all five required safety policies
  (always-answer 0/50 pairs, always-refuse 0/50, correct selective refusal 50/50,
  malformed 0/50, degenerate 0/50).
- **Status: frozen.** Changing any rule requires bumping `BATTERY_VERSION`.
- **License/provenance:** derived from the public sources above; no teacher
  generations, no user data.
- **Related logs:** [`PROPOSAL.md`](archive/PROPOSAL.md) §6–§7,
  [`EXPERIMENTS.md`](EXPERIMENTS.md) §12.7–§12.8.

## checkpoint_inventory — both stores, 2026-08-03

- **Artifact:** `artifacts/stage3/checkpoint_inventory.json` — every weight and
  optimizer-state file on the dev box and in `AlphaAvatar/aadistill-artifacts`,
  with size, hash, duplicate status, required-by and proposed action.
- **Totals:** dev box 9 files / 17.51 GiB before cleanup, 7 / 13.32 GiB after;
  relay 34 files / 73.28 GiB, untouched.
- **Deleted:** the two-step ladder smoke test's `model.safetensors` (2.22 GiB)
  and `trainer_state.pt` (1.97 GiB). **4.19 GiB reclaimed**, dev box 117 → 121
  GiB free. Its manifest, train log and model config were kept.
- **Duplicates found:** exactly two — the Stage 1 `checkpoint` and
  `random_baseline`, byte-identical between dev box and relay by LFS object
  sha256. Every other checkpoint on either store is single-copy.
- **Relay: 0 bytes reclaimed and no file touched.** Ordinary deletion does not
  free LFS quota there; the operations that would all invalidate existing
  revisions and are reported for separate approval.
- **Related logs:** [`PROPOSAL.md`](archive/PROPOSAL.md) §9.

## Experiment 2 phase 1 outputs (2026-08-04)

Pod `n7xjbzlmsyx9b2` deleted after transfer; nothing remains on it.

### Results bundle — transferred and hash-verified

| field | value |
| --- | --- |
| file | `e2p1_results.tar.gz` |
| size | 2,676,197 B |
| sha256 | `b70e2ffb8efa59a3520a9781b6daa8e958e45cfec775c1c9f32940dd6aeee6be` |
| verified | byte-identical on the pod and after transfer |
| location | dev box, session scratchpad + unpacked alongside the checkpoints |
| creation | `tar czf … eval/e2p1 --exclude=checkpoints stage3/e2_d1_{sa,sb}_pca` |

Contents: complete raw generations for all 846 prompts × 6 full-battery
checkpoints plus 76 prompts × 14 behaviour-only points, per-sample verdicts, 9
battery scorings, 20 `behavior_v0` measurements, both `holdout_trajectory.jsonl`,
both `run_manifest.json`, both `train_log.jsonl`, `throughput_gate.json`, and the
same-machine D0 control `d0_holdout_nll_samemachine.json`.

### Retained checkpoints — dev box only, NOT the relay

`/home/ecs-user/aad-artifacts/e2p1/`, 23.7 GB, 7 checkpoints. Kept per the
pre-registered retention rule; the two `step_001023` entries include
`trainer_state.pt` and are resumable.

| arm | step | identities | size |
| --- | --- | --- | --- |
| `e2_d1_sa_pca` | 508 | `best_holdout_nll`, `deterioration_onset` | 2.3 G |
| `e2_d1_sa_pca` | 635 | `after_deterioration_onset` | 2.3 G |
| `e2_d1_sa_pca` | 1016 | `best_val_ce` | 2.3 G |
| `e2_d1_sa_pca` | 1023 | `final` (+ trainer state) | 5.6 G |
| `e2_d1_sb_pca` | 127 | `best_holdout_nll`, `deterioration_onset` | 2.3 G |
| `e2_d1_sb_pca` | 254 | `after_deterioration_onset` | 2.3 G |
| `e2_d1_sb_pca` | 1023 | `final`, `best_val_ce` (+ trainer state) | 5.6 G |

Dropped by retention on the pod before transfer: 24.6 GB across the 11
non-retained eval points, whose **generations and metrics were all captured
first** — every eval point has a `behavior_v0` measurement in the bundle.

**Not uploaded to the relay.** Its LFS quota is full and ordinary deletion does
not reclaim it, so these live on the dev box (117 GB free before, ~93 GB after).
An upload would need either a new repository or an approved history rewrite;
neither was requested and neither is needed, since no follow-up arm starts from a
trained checkpoint — the maintainer's standing rule is that every new arm forks
from the Stage 1 init (`86fbba78e8a2a324…`).

## Diagnostic session outputs (2026-08-04)

Pod `tct4820z4t3hvn` (RTX A6000, $0.33/h) deleted after transfer; nothing remains.

| field | value |
| --- | --- |
| file | `e2diag_results.tar.gz` |
| size | 2,101,323 B |
| sha256 | `5988da19e01dfcc697ecc9604591b4a4032ded67b6acb4e4b646c0f048f0a290` |
| verified | byte-identical on the pod and after transfer |
| unpacked to | `artifacts/audit/`, `artifacts/eval/e2diag/` (gitignored) |

Contents: `padding_truncation_benchmark.json` (4 regimes × 2 paths, per-regime
s/step, peak memory, student/teacher forward split); `reference_geometry.json`;
`eval/e2diag/ref_qwen3_0p6b_{project,native}/` with complete raw generations for
846 prompts each plus their battery scorings and per-sample verdicts;
`audit/training_recall/` with 429 generations (now including token ids),
`gold_prefix_top1.jsonl` and `report.json` (generations sha256 `479df460…`); and
`audit/training_recall_specialstripped/` — the first pass, retained because its
`skip_special_tokens=True` decode is what revealed the scoring bug.

Model identities pinned in every record: `Qwen/Qwen3-0.6B @ c1899de2…`,
`Qwen/Qwen3-4B-Thinking-2507 @ 768f209d…`, chat-template sha256, thinking-mode
setting, and library versions (torch 2.11.0+cu128, transformers 5.13.1,
vLLM 0.26.0).

The control checkpoint `e1_ctl_r0250k_sa_pca_stepmatched` was pushed from the dev
box (`model.safetensors` sha256 `bfdcb4436f51eb31deea810be45a20e1fd39f6614d4f4b78c60ab952529858e2`,
2,384,234,968 B) and verified on the pod by hash before diagnostic B ran. It
remains dev-box-only; the relay holds only its evaluation JSONs.

## D0 diagnostic outputs (2026-08-04)

Pod `0cn6ipb4aobca3` (RTX A6000, $0.33/h) deleted after transfer; nothing remains.

| field | value |
| --- | --- |
| file | `d0_final.tar.gz` |
| sha256 | `2a29c4fc2938ed34c97e21efb5aab72858df027a42ffd11766a987af16ea6960` |
| verified | byte-identical on the pod and after transfer |
| unpacked to | `artifacts/audit/three_mode/`, `artifacts/audit/kd_decomposition_*.json` |

An earlier partial bundle `d0_partial.tar.gz` (sha256 `21eebf14976619ea…`) was
pulled mid-run so no completed result depended on the pod surviving.

Contents: `three_mode/P0-real-{sa,sb}/` with 150 free and 150 oracle generations
each (token ids included), `forced.per_sample.jsonl`, and the recomputed
`summary_free_oracle.json`; plus `kd_decomposition_P0-real-{sa,sb}.json` over the
full 682-block rung, batch sha256 `bffd9305fe5d51ef…`.

Registration: `artifacts/audit/p0_real_registration.json`. Inclusion mask
`d6e24e0b09da1bcc692b1dc96d8236808d29551a9fc94a47d1d968fd3f73d6ba`.

## P0-assistant outputs (2026-08-05)

Pod `4pge0934xnly22` (L40S secure, $0.99/h) deleted after transfer.

| field | value |
| --- | --- |
| file | `p0asst_final.tar.gz` |
| sha256 | `9c2ad23e292b8857623a5379ec13462b5eec8b650560ce81be316720a21c3321` |
| verified | byte-identical on the pod and after transfer |

Contents: `three_mode/P0-assistant-{sa,sb}/` with 150 free and 150 oracle
generations each plus the teacher-forced per-role reports, and both arms'
`train_log.jsonl` / `run_manifest.json`.

Configs: `configs/stage3/p0/p0_assistant_sa.json` (`dccf60d0f623a3f2…`),
`p0_assistant_sb.json` (`252f09463773add1…`).

**The two P0-assistant checkpoints were left on the pod and are gone.** They lost
on the pre-registered selector, `P1` aliases the P0-real arms instead, and
retaining 4.6 GB of rejected weights was not worth the transfer time. Their
training logs, manifests and complete evaluation generations are retained, so the
result is fully reproducible from the configs.

## P2-ceheavy outputs and checkpoints (2026-08-05)

Pod `r3dlq1g6q51xnw` (L40S secure, $0.99/h) deleted after transfer and
verification. Store: `/home/ecs-user/aad-artifacts/p2_ceheavy/` (external to git;
`artifacts/` is `.gitignore`d).

### Checkpoints — retained, unlike P0-assistant

Both `step_001023/model` directories transferred and verified with
`sha256sum -c` against a manifest computed **on the pod before transfer**: 12/12
files OK. 2.3 GB each.

| file | p2_ceheavy_sa | p2_ceheavy_sb |
| --- | --- | --- |
| `model.safetensors` | `4aface45a12cd02e…` | `9828b1780a5eb4e2…` |
| `config.json` | `62d14acfb86e397e…` | *(identical)* |
| `generation_config.json` | `0019fccc989feeeb…` | *(identical)* |
| `chat_template.jinja` | `3802169b2a02b81e…` | *(identical)* |
| `tokenizer.json` | `be75606093db2094…` | *(identical)* |
| `tokenizer_config.json` | `8fa82a4ba512c8be…` | *(identical)* |

The four tokenizer/config files are byte-identical across both arms **and** to
`artifacts/stage1/qwen3_0p6b_init_v0/checkpoint`, which is where the driver
copied them from. Full manifest: `p2_ckpt.sha256` in the store.

Both load on CPU: 596,049,920 params, RoPE base 4,999,984, finite logits.
`trainer_state.pt` was **not** retained — exact training resume is not needed.

These arms also lost on the pre-registered selector, but were retained on
explicit instruction, so unlike P0-assistant the weights remain measurable.

### Side artifacts

| field | value |
| --- | --- |
| file | `p2_side.tar.gz` |
| sha256 | `7e95040b503c149dbf9163dba9c3ffca6e3cbbb2a6a8bc27b0c0083b026e3f09` |
| size | 532 KB, 24 entries |
| verified | byte-identical on the pod and after transfer |

Contents: `artifacts/audit/p2_holdout_nll.json`;
`artifacts/audit/three_mode/P2-ceheavy-{sa,sb}/` with 150 free and 150 oracle
generations each (token ids included), `forced/forced.per_sample.jsonl` and both
reports; `configs/stage3/p2/p2_ceheavy_{sa,sb}.json`; both `run_manifest.json`
and `train_log.jsonl`. Pod `p2_run.log` (107 KB) and `p2.status` retained beside
it.

Configs, canonical `sha256_json` of the parsed config as recorded in the run
manifests: sa `42616c1921419d01…`, sb `b846fee7bcae670f…`. These intentionally
differ from `sha256sum` of the file bytes (`a43775013a5534b5…`, `01b6c3cfa47a7e4e…`)
— the trainer hashes the parsed config so formatting cannot change the identity.

Inclusion mask (shared with P0-real and P0-assistant):
`d6e24e0b09da1bcc692b1dc96d8236808d29551a9fc94a47d1d968fd3f73d6ba`.


## Experiment 1 checkpoint recoverability — verified 2026-08-05

Full detail in [`EXPERIMENTS.md`](EXPERIMENTS.md) §19.14. Digests:
`artifacts/audit/relay_e1_digests.json` (git-ignored; regenerate from the relay).

* **Local** `artifacts/stage3/rescued/`: 30/30 files verified against the
  pod-side `pod_hashes.txt` recorded before transfer — 0 mismatched, 0 missing.
  Covers `e1_r2960k_sb_{pca,rand}`, `e1_r5500k_sb_{pca,rand}` and the 0.25M
  step-matched control.
* **Relay** `AlphaAvatar/aadistill-artifacts` (729 files): `e1_r0860k_{sa,sb}_pca`
  (**= P1**), all four 1.60M arms, and `e1_r2960k_sa_{pca,rand}` are present at
  2.38 GB each with LFS sha256 recorded.
* **End-to-end verified:** `e1_r1600k_sa_pca` (`6f77676ab8fde397…`) and **P1-sa**
  (`18ee10a10333481d…`) downloaded, ~165 s each, recomputed sha256 matching the
  LFS digest exactly. The relay path is live and its digests are real.

### P1 rescued to the dev box — 2026-08-05

**Resolved.** P1 no longer exists in one place only. Both arms were copied from
the relay to `artifacts/stage3/rescued/` and **hash-verified against the relay LFS
digests**:

| arm | `model.safetensors` sha256 | verdict |
| --- | --- | --- |
| `e1_r0860k_sa_pca` (P1-sa) | `18ee10a10333481d…` | **matches relay LFS digest** |
| `e1_r0860k_sb_pca` (P1-sb) | `f66de5320b69aa34…` | **matches relay LFS digest** |

Both load on CPU: 596,049,920 params, RoPE base 4,999,984, finite logits. Each
directory carries a full tokenizer (the trainer's `save_checkpoint` does not write
one) and a `pod_hashes.txt` for future verification. 4.6 GB; 80 GB free remains.

The risk this closes: the relay is at its private-storage limit and has an
approved-but-unexecuted history squash pending, **which invalidates existing
revisions**. P0-assistant's weights were already lost to the same class of
problem. Every Stage 2/3 candidate that still has weights now has a verified local
copy.

### `e1_r5500k_sa_pca` — digest recorded 2026-08-08, at E6 preflight

The 2026-08-05 sweep recorded relay LFS digests for eight E1 arms; the 5.50M `sa`
pair was not among them, so the top rung of the PCA curve had **no recorded
weight hash anywhere** — not locally, not in `relay_e1_digests.json`, and not in
its own `pod_hashes.txt`, which was never retained for the relay-only arms.

Recorded now, from the relay's LFS metadata and independently confirmed by
recomputation on the E6 pod after download:

| arm | path | size | sha256 |
| --- | --- | ---: | --- |
| `e1_r5500k_sa_pca` | `e1_scaling_20260801/e1_r5500k_sa_pca/step_004412/model/model.safetensors` | 2,384,234,968 | `3069b329df3edfbd5edc0356516cd06ee7f02fe59663c19df7b30ef6acd8e397` |

**Provenance boundary, stated rather than papered over.** This digest pins the
bytes the relay holds at the canonical Experiment 1 path, step and size, and E6
evaluated exactly those bytes. It is **not** an independent attestation from the
training pod, because none was retained for this arm — unlike the four
`rescued/` arms, whose `pod_hashes.txt` was written before transfer. A future
session should treat it as "the artifact the relay published" rather than "the
artifact the trainer wrote", and the two are only known to coincide by the
absence of any event that would separate them.

## Experiment 3 outputs and checkpoints (2026-08-05)

Pod `94slla57nnqjqa` (L40S secure, $0.99/h) deleted after transfer and
verification. Store: `/home/ecs-user/aad-artifacts/e3/` (external to git;
`artifacts/` is `.gitignore`d). **28/28 files `sha256sum -c` OK** against
`e3_pod_hashes.txt`, computed **on the pod before transfer**.

### Checkpoints — 18.6 GB, dev box only

| arm | contents | size |
| --- | --- | ---: |
| `e3_a1_frozen_attn_sa` | `model/` (merged HF) + `trainer_state.pt` | 4.3 GB |
| `e3_a1_frozen_attn_sb` | same | 4.3 GB |
| `e3_a2_lora_attn_sa` | `model/` (merged HF) + `lora_state.safetensors` (741 MB) + `trainer_state.pt` | 5.0 GB |
| `e3_a2_lora_attn_sb` | same | 5.0 GB |

Every A2 checkpoint is **both deployable and resumable**: `model/` is a plain
Hugging Face checkpoint with the LoRA delta merged into q/k/v/o and **no adapter
keys**, verified to load through the normal path with no LoRA code; the frozen
base attention weights and raw A/B tensors live in `lora_state.safetensors` for
exact resume. The merged delta was independently re-derived as
`scaling · B @ A` and matched on both arms.

The four tokenizer/config files are byte-identical across all four arms **and**
to `artifacts/stage1/qwen3_0p6b_init_v0/checkpoint`, which is where the driver
copied them from.

**Not uploaded to the relay.** Its LFS quota is full, ordinary deletion cannot
reclaim it, and no follow-up arm forks from a trained checkpoint — every arm
forks from the Stage 1 init `86fbba78e8a2a324…`.

### Side artifacts

`e3_side.tar.gz`, 1.2 MB: all four arms' 150 free + 150 oracle generations with
full token ids, teacher-forced per-role reports, `e3_movement/` (six arms),
`e3_merge_check.json`, both pre-launch validations, the four E3 configs, and each
arm's `run_manifest.json` / `train_log.jsonl`. Also
`e3_aborted_a2_sa_alpha16/` — the log and manifest of the α=16 run stopped at
step 180 of 1,023, retained as a record and **not** a data point.

Alongside: `e3_run.log` (232 KB), `e3.status`, `e3_pod_hashes.txt`.

Configs: `e3_a1_frozen_attn_sa` (`cc6ba2972a28c7c2…`), `_sb` (`477cbf347ba39df4…`),
`e3_a2_lora_attn_sa` (`f9c5cdd26af067ae…`), `_sb` (`74e8846e25b12d1f…`) —
`sha256_json` of the parsed config, matching `logs/e3_registration.json`.


## Experiment 4 outputs and checkpoints (2026-08-06)

Pod `qzevis6g43en33` (L40S secure, $0.99/h) deleted after transfer. Store:
`/home/ecs-user/aad-artifacts/e4/`, **12/12 files `sha256sum -c` OK** against
`e4_pod_hashes.txt`, computed on the pod before transfer.

| arm | contents | size |
| --- | --- | ---: |
| `e4_p2_r1600k_sa` | `model/` (step_001761) + `trainer_state.pt` | 5.6 GB |
| `e4_p2_r1600k_sb` | same | 5.6 GB |

Config hashes `8256bfba8b3241a8…` (sa) / `7c3817a729133dc9…` (sb), matching
`logs/e4_registration.json`. Both fork from the Stage 1 init `86fbba78…`; neither
continues from P2-0.86M.

`e4_side.tar.gz` (1.1 MB): 150 free + 150 oracle generations for all four
evaluated checkpoints (both new arms **and** the two P1-1.60M re-evaluations),
teacher-forced per-role reports, `e4_movement/`, the preflight, both configs and
each arm's `run_manifest.json` / `train_log.jsonl`.

**Not uploaded to the relay** — LFS quota is full and no follow-up arm forks from
a trained checkpoint.

**The P1-1.60M re-evaluation is the reusable asset here**: those two checkpoints
now have 150-example unrestricted-harness results (`three_mode/P1-1600k-{sa,sb}/`)
comparable with every Stage 2/3 family, where before they had only 76-prompt
behaviour-wave numbers taken with the degeneration stop active.

## Experiment 3 checkpoints — DELETED 2026-08-09 (maintainer-approved)

- **Artifact:** the four Experiment 3 arms, `e3_a1_frozen_attn_{sa,sb}` and
  `e3_a2_lora_attn_{sa,sb}`, formerly at `/home/ecs-user/aad-artifacts/e3/`,
  **19.6 GB**, dev box only (never on the relay — its private LFS quota has been
  at the limit since 2026-08-02).
- **Why deleted:** Experiment 3's verdict is that restricting attention updates
  **degrades** autonomous generation stability (−0.087 and −0.093 usable rollout,
  both seeds, every component). **Neither arm was adopted** and no live claim
  depends on the weights. Deleted to reclaim dev-box space for training sessions.
- **Weight hashes, recorded before deletion** so the record identifies exactly
  what existed:

| file | sha256 |
| --- | --- |
| `e3_a1_frozen_attn_sa/model/model.safetensors` | `c813b5972e7c140103880fb50b800e9ba4a617d4c15215d717c64b77c640532c` |
| `e3_a1_frozen_attn_sb/model/model.safetensors` | `104813dec07cf9acfa11cb8103bb491de520d1338979326c021bf0f7f35369a0` |
| `e3_a2_lora_attn_sa/model/model.safetensors` | `a8f397791b2b94c2d834c158494efa99733c71e67219b98bdb5471b4e2b5bb20` |
| `e3_a2_lora_attn_sa/lora_state.safetensors` | `34f9ab5d7d2af6920d72151c4fcd173a254bea092b43924e4bbe6c17c5869f70` |
| `e3_a2_lora_attn_sb/model/model.safetensors` | `2e6530d05b2ab59672974b5cb420d3ac7fab9b0bc2299291b67616f2c7f61100` |
| `e3_a2_lora_attn_sb/lora_state.safetensors` | `6eb5132291706a125ee0904126e7afd64ad056c354e460df5e1bfb48b739f04a` |

- **What survives, and it is everything except the weights:** all four evaluation
  sets (`artifacts/audit/three_mode/A{1,2}-*`, 150 generations each with token
  ids and decoded text), `artifacts/audit/e3_comparison.json`,
  `artifacts/audit/e3_movement/`, the preregistration
  [`logs/e3_registration.json`](e3_registration.json), and the full record at
  [`EXPERIMENTS.md`](EXPERIMENTS.md) §20. **The experiment remains fully
  re-analysable; it is not re-runnable without retraining** (~$5.76).
- **Distinguish this from a loss.** P0-assistant's and E5's checkpoints were
  *lost* to defects. These were *deleted deliberately*, with hashes recorded and
  maintainer approval, because a rejected approach's weights are the cheapest
  thing on the disk to give up.

## Experiment 6b outputs and checkpoints (2026-08-09)

Session commit `6375e299815416dddc1bd0c12fd6fe273035a9e9`, pod
`luy1txyjro2msz` (L40S, $0.99/h, 458 min).

### Checkpoints — dev box only, NOT the relay

The relay's private LFS quota has been at its limit since 2026-08-02 and
deletion does not reclaim it, so these were never uploaded. Both were fetched
from the pod before teardown and verified local-vs-pod.

| arm | path | `model.safetensors` sha256 | verified |
| --- | --- | --- | --- |
| `e6b_p2_r2960k_sa` | `/home/ecs-user/aad-artifacts/e6b/e6b_p2_r2960k_sa/` | `89b14b839ff9b8a2e4651dbfaee63ab2703cd5737c11af9afb438dc2599497e7` | **matches pod manifest** |
| `e6b_p2_r2960k_sb` | `/home/ecs-user/aad-artifacts/e6b/e6b_p2_r2960k_sb/` | `3c4709b51792c7e6e18c512c240bb20f6b55b0714e6d2c1264522e30633856f6` | **matches pod manifest** |

5.6 GB each, `step_002916`, each carrying a full tokenizer (the trainer's
`save_checkpoint` writes none) and dressed from the Stage 1 init
`86fbba78…`. Pod-side manifest retained at
`/home/ecs-user/aad-artifacts/e6b/e6b_ckpt_hashes.txt`.

**Single-copy risk, recorded.** Neither arm exists anywhere but this dev box.
Reproducing either costs ~200 min of L40S (~$3.30). The same class of loss took
P0-assistant's weights permanently and all four E5 checkpoints.

### Evaluation artifacts — retained and committed in summary form

| artifact | where | note |
| --- | --- | --- |
| raw generations, both arms | `artifacts/audit/three_mode/P2-2.96M-{sa,sb}/` | gitignored; free + oracle + forced, 150 prompts each, mask `d6e24e0b…` |
| retrieved bundle | `/home/ecs-user/aad-artifacts/e6b/e6b_artifacts.tar.gz` | sha256 `d96d63a97082af70…`, digest-verified before teardown |
| driver console log | `/home/ecs-user/aad-artifacts/e6b/e6b_run.log` | 132 KB; carries the training curve, per-step times and final val CE |
| summary + report | `logs/e6b_results.json`, `logs/e6b_report.md` | tracked; reproduce byte-identically from the generations |
| per-prompt records | `artifacts/audit/e6b_per_prompt.jsonl` | 1,200 scored rows |

### NOT retained — a P4 gap

`train_log.jsonl` and `run_manifest.json` for **both** arms were never fetched.
**Corrected attribution, 2026-08-09:** this entry previously blamed "the pod-side
bundling command's `$(ls -d …)` globs [that] did not expand inside the ssh
quoting". That is wrong — the E6b bundling command contains no glob. Its path
list was inherited verbatim from E6, a session that did not train, so the event
streams were **never listed**; `tar tzf` on the retrieved bundle confirms it
holds `artifacts/audit/three_mode/**` and nothing else. Every downstream check
then passed on the incomplete bundle. Full record:
[`e6b_protocol_deviations.md`](e6b_protocol_deviations.md).

The training curve, per-step timings and final validation CE survive in the
driver console log, so the substance is recoverable; the machine-readable event
stream required by AGENTS.md 3.7 is not.

**Derived artifact:** `logs/e6b_reconstructed_training_events.json`
(`provenance: reconstructed_from_driver_console`,
`original_event_stream_available: false`) — 291 `train_step` + 10 `eval_result`
events per arm, with a per-field provenance block. **Not** the original stream.

**Fixed, 2026-08-09:** collection is now manifest-driven
(`scripts/pod/collect_artifacts.py`), the archive is built from the manifest
rather than from a shell glob, and a missing required artifact class blocks
teardown (EXPERIMENTS.md §30).

## E7 extra-KD streams — built 2026-08-09, CPU, $0 (dev-box only, NOT yet on the relay)

Gitignored under `artifacts/stage3/`; each carries `manifest.json` + `docs.jsonl`
with per-document sha256. **They must be staged to the relay before any E7 pod
session** — they are currently single-copy on the dev box.

| stream | kind | blocks x len | KD positions | `blocks.npz` sha256 |
| --- | --- | --- | ---: | --- |
| `e7_fineweb_kd` | general-text KD (arm B) | 1761 x 1024 | 1,801,503 | `b70beffac337ee37b0280cef581fec7967a1d5d6a390a73272d75223aeb39633` |
| `e7_control_kd` | matched in-domain KD control (arm C) | 1761 x 1024 | 1,801,503 | `4e54f8e18baf01dc44eee80ae044188e902fa5f2ce675723ccdc5bf3aa5c05ea` |
| `e7_fineweb_val` | general-text validation | 512 x 1024 | 523,776 | `e4002bbbbadf1a9106f41e2531fffddcf9ce329b894232246211b9c5d665014e` |

Sources. **FineWeb**: `HuggingFaceFW/fineweb-edu`, config `sample-10BT`, split
`train`, revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, ODC-By 1.0; train
index range [30000, 31902), validation [20000, 20454); `doc_char_min` 500, no cap;
raw text, no chat template, zero CE positions; `<|endoftext|>` document
separator; partial tail dropped, so zero padding. **Control**: content tokens of
canonical-pack blocks [1174, 1853) — after the 1.60M rung, before the validation
tail at 2941 — re-packed densely under the identical boundary policy; pack
`blocks.npz` sha256 `6f324cb0f37bc0f0…`.

Disjointness: `artifacts/stage3/e7_disjointness.json` — index ranges **and**
content hashes, against `holdout_v1`, `warmup_v1`, `eval_behavior_v0/prompts.jsonl`
and all seven `capability-v2` files. Zero overlaps involving any E7 stream.

Rebuild commands: [`e7_preregistration.md`](e7_preregistration.md) §11.



## E7 arms — trained 2026-08-09, $10.49 (dev-box only, NOT yet on the relay)

Four checkpoints at `step_001761`, 5.6 GB each, under
`/home/ecs-user/aad-artifacts/e7/`. Per-file sha256 recorded pod-side before
transfer and re-verified locally; the session's 38-file artifact bundle verified
with **zero hash problems** (37 `final_required` + 1 `mutable_snapshot`,
`final_streams_quiescent: True`).

| arm | extra stream | extra KD positions |
| --- | --- | ---: |
| `e7_fineweb_r1600k_sa` | FineWeb-Edu raw text | 1,801,503 |
| `e7_fineweb_r1600k_sb` | FineWeb-Edu raw text | 1,801,503 |
| `e7_control_r1600k_sa` | matched in-domain KD-only | 1,801,503 |
| `e7_control_r1600k_sb` | matched in-domain KD-only | 1,801,503 |

All four fork from the Stage 1 PCA init (`86fbba78…`), train the identical 1.60M
rollout stream, and differ only in the extra stream's content. Retained
evaluation artifacts: `artifacts/audit/three_mode/E7-*` (generations, forced
per-sample, reports), `artifacts/audit/e7_general_text/` (six models incl. the
retained arm A), `artifacts/audit/e7_per_prompt.jsonl` (900 scored records),
`artifacts/audit/e7_movement/`.

**Not on the relay.** Single-copy on the dev box. The E7 *inputs* (streams,
disjointness proof, holdout) are on the relay under `e7_streams_20260809/`; the
produced weights are not, and re-evaluating an arm without them would cost a
retrain.

## vLLM wheelhouse — built 2026-08-15, local copy retired 2026-08-18

- **Artifact:** `AlphaAvatar/aadistill-artifacts`, prefix
  `transfer/wheelhouse_vllm_cp312/`. 196 wheels, 3,886,432,359 bytes.
- **Manifest:** `wheelhouse_vllm_sha256.json` in the repository root — tracked,
  and it freezes the wheel **bytes**, not only the pinned versions:
  `manifest_sha256 f9c0e814b4b93323…`, `n_wheels 196`, one sha256 per file. The
  pod verifies every entry after fetching and refuses to install on any
  mismatch, so a silently re-uploaded or truncated wheel cannot reach a paid run.
- **Creation command:** `scripts/pod/build_wheelhouse.py --from-pins
  --requirements requirements-vllm.txt`; each wheel was verified against the
  PyPI sha256 for its pinned version at download time.
- **Interpreter/platform:** cp312 / manylinux x86_64.
- **Licenses:** the wheels' own, unmodified; nothing is repackaged.
- **Local copy:** was `/home/ecs-user/aad-artifacts/wheelhouse_vllm_cp312`
  (3.620 GiB), the staging copy left after upload. **Retired 2026-08-18** —
  tombstone `wheelhouse_vllm_cp312_local_cache`. All 196 files were verified
  byte-identical to the relay first: 123 by LFS oid, 73 downloaded and hashed,
  recorded in [`relay_mirror_verification.json`](relay_mirror_verification.json).
  Pods fetch from the relay, never from the dev box, whose uplink is 0.72 MB/s.
- **Related:** `logs/decisions.md` 2026-08-18 (cleanup);
  `requirements-vllm.txt`; `docs/POD_SCRIPTS.md` (`build_wheelhouse.py`).

## Preserved sole-copy Phase-A/B raw evidence — promoted 2026-09-01

- **Artifact:** `/home/ecs-user/aad-artifacts/autoinit/preserved_scratch_20260901/`
  — out-of-tree, **not** in git. 359 files, 13,575,205 bytes (12.95 MiB).
- **Manifest:** `MANIFEST.json` inside that directory, schema
  `aadistill.promoted_raw_evidence/v1`, **sha256
  `3e1a72e2ade2fde610b95e41e147f7f93d37fd4418a4a2d9f61ae58d27cca0b7`**
  (156,532 bytes). One record per file: `origin` (the `aad-scratch` path it was
  copied from), `path` (relative layout, preserved under the originating session
  directory name), `bytes`, `sha256`, `role`, `verified`.
- **Contents by scientific role:** 12 per-sample row files · 90 raw generation
  files · 24 training event streams · 12 scored probe results · 18 probe configs
  · 1 evaluation-protocol attestation · 102 run-metadata files · 99 session logs
  · 1 other.
- **Why it exists:** the Phase-C0 audit found that **11 of the 12** retained
  Phase-A per-sample row files, and 90 raw generation files, were **sole copies**
  under `aad-scratch` session directories outside the declared
  `aad-scratch/sessions/<session-id>` convention. AGENTS.md P18 requires the
  complete raw generation to be retained for every evaluated sample, and every
  `$0` variance, ICC and paired analysis behind the frozen Phase-C0 design rests
  on these rows.
- **Creation command:** a read-only inventory pass (every `aad-scratch` file
  size-indexed against repo `logs/`, repo `artifacts/` and `aad-artifacts`, with
  sha256 computed only inside size-collision groups) followed by `shutil.copy2`
  of every sole copy under a session `store/extracted/` tree, each verified by
  re-hashing source and destination.
- **Verification:** all 359 re-hashed from the manifest — **0 missing, 0
  size/hash mismatch**; **359/359 originals still present**; and the preserved
  copy *alone* re-derives the frozen Phase-A/B pooled figures — `cca699c93f34`
  15/510 = 0.029412 (usable 0.6561), `85bde4ded2c3` 10/510 = 0.019608 (usable
  0.5456), control 3/340 = 0.008824 (usable 0.4947).
- **Copy, not move.** The `aad-scratch` originals are **retained** and must not
  be deleted in this phase.
- **Why a new sibling directory:** the three probe-reuse verifiers all read
  `aad-artifacts/autoinit/phase_a`, and the attempt-5
  `raw_evidence/MANIFEST.json` enumerates a fixed 89-file set. A new directory
  disturbs neither.
- **Excluded deliberately:** 140.66 MiB of session transport bundles
  (`*.bundle`) and store tarballs — operational transport, not scientific
  evidence. No decision on them is required in C0.
- **Known gap:** per-sample rows for `fe9683e6a9c7/sb` do not exist anywhere;
  only the aggregate probe JSON survives. They are **not** reconstructed from
  aggregate counts.
- **Related:** `logs/decisions.md` 2026-09-01; `logs/current_state.json`
  (`storage.preserved_scratch_20260901`); `logs/STATE.md`;
  `logs/phase_c0_sizing_evidence.json`.

## Phase-C1 confirmation battery — built and frozen 2026-09-01

- **Artifact:** **canonical at
  `/home/ecs-user/aad-artifacts/autoinit/c1_confirmation_v1/`**, with the
  repo-local working copy at `artifacts/stage3/c1_confirmation_v1/`. Both are
  out of tree — `.gitignore` excludes `artifacts/`, exactly as it does the frozen
  `recovery_search_v2` — so the battery is identified by hash rather than stored
  (AGENTS.md 2.5). 950 prompts / 850 scorable, 3.26 MiB, 8 files.
- **Canonicalized 2026-09-02 by COPY, not rebuild.** The frozen bytes were
  promoted unchanged and re-verified *from the copy*: every file byte-identical,
  every set sha256 equal to the frozen manifest value, `content_sha256`
  re-derived to `a285d61f…`, membership 950/850. The repo-local copy is
  retained. The scientific identity and the selection rule are untouched.
- **Committed identity:** [`phase_c1_battery.json`](phase_c1_battery.json) —
  asset id, per-set sha256, pinned source revisions, sampling rule, isolation
  result.
- **Content hash:**
  `a285d61f88de9da85e87818786cce8d350f03246365ff946207c61a6464fee3c`, over
  newline-joined sorted `id:prompt_sha256` pairs — the same convention
  `recovery_search_v2` uses, so equality proves membership, ordering, ids and
  prompts are all fixed.
- **Mixture:** gsm8k 150 · math_verified 150 · multihop 150 · rag 150 ·
  knowledge 150 · tool 100 · code 100 (behaviour-only). The frozen historical
  3:3:3:3:3:2 ratio scaled ×5, **not** reweighted by historical correctness.
- **Selection:** ascending `SHA256(C0_digest + ":phase-c1-battery:" + stratum +
  ":" + stable_source_id)`, ties by ascending id. The key is fixed by the pushed
  C0 preregistration digest `fb2eeea5…`, which predates every C1 candidate. **No
  model output of any kind is consulted**, and no source-native difficulty field
  stratifies the sample.
- **Creation command:** `PYTHONPATH=src .venv/bin/python
  scripts/data/build_c1_confirmation_battery.py`. Renderers shared with the
  recovery-search convention via `scripts/data/battery_render.py`.
- **Determinism:** an independent rebuild into a separate directory reproduced
  `content_sha256` and every per-set sha256 exactly. `manifest_sha256` differs
  between builds only because it embeds `created_utc` and the output path.
- **Isolation:** `scripts/autoinit/verify_c1_battery_isolation.py` — **PASS**,
  0 stable-id and 0 normalized-content collisions against each of
  `FINAL_PROMOTION`, `RECOVERY_SEARCH`, the full recovery-training corpus,
  `STATE_EVALUATION` and `OPERATOR_CALIBRATION`.
- **`FINAL_PROMOTION` is read for exclusion only.** It is not sampled from and
  remains unconsumed as an evaluation asset.
- **Sources:** the same pinned snapshots `recovery_search_v2` was built from —
  gsm8k `740312ad`, MATH-500 `6e4ed1a2`, HotpotQA `1908d6af`, SQuAD-v2
  `3ffb306f`, TriviaQA `0f7faf33`, xLAM `26d14ebf`, MBPP `4bb6404f`. The two
  batteries differ in their sample, never in their source.
- **Status: no model has been evaluated on it.** It exists so the C1 execution
  preregistration can bind it; nothing has been measured.
