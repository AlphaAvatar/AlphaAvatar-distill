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
  ([`PROPOSAL.md`](PROPOSAL.md) §2); the 4,096-token cap censored 19.9% of
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

## corpus_v2_clean + ladder_uniform_clean_anchored — Experiment 2 phase 1 (D1)

- **Artifact:** `artifacts/stage3/corpus_v2_clean/` (`sessions_clean.jsonl`
  sha256 `7d22b3e037b0de45c2b7fd23d9fb6706955eef59f472d9e6937b8e4ea1a76bc3`,
  68 MB; `cleaning_audit.json` `8d97e03a5aadbee40b17aa276bf89a8a805cee275006d6fa2f2c7cb3b541a324`;
  `cleaning_per_example.jsonl`; `d0_session_order.txt`) and the pack
  `artifacts/stage3/ladder_uniform_clean_anchored/` (`blocks.npz`
  `1c8792dbc796e22f5547e0848ad7bd301b5c0b1a8a369672efc05ddce5e37b79`,
  `ladder.json` `510476400f661a351b50352a7744a2736b388247c8edfc42a10b7d8383ae9d05`,
  `audit.jsonl` `213a422d2e690abd0fe04a4cb295e69895ae650e457236fec2d9eaa62c7130ef`).
  Comparison record: `artifacts/stage3/e2_d1_corpus_audit.json`
  `a27ae3dfe755a2fb081a42e87720e7071bac702e84dcd5a888ad3929176659bb`.
- **Created:** 2026-08-03, **CPU only, $0**. 114 s to screen 11,174 sessions,
  ~9 min to pack. No teacher inference: every target is a completion already
  present in corpus v2's retained `n=4` candidates.
- **Derivation:** `scripts/data/build_cleaned_corpus.py` (rules
  `aadistill.data.cleaning`, `RULES_VERSION = "clean-v1"`) over
  `candidates.jsonl` `f7f5035e…` + `sessions.jsonl` `2b4edc2e…`, then
  `scripts/data/build_token_ladder.py --session-order` anchored to Experiment
  1's pack, uniform mixture, block-len 8192.
- **Contents:** 10,778 of 11,174 sessions (96.5%); the 2,968,828-supervised-token
  rung is 1,944 blocks — block-, step- and packed-token-identical to Experiment
  1's 2.96M rung.
- **Tokenizer/template:** teacher `Qwen/Qwen3-4B-Thinking-2507@768f209d`, vocab
  sha256 `3ec3c124…`, chat template `3802169b…` — both reproduced exactly on the
  dev box under transformers 5.13.1 against the corpus's 5.14.1.
- **Status: prepared, NOT trained on.** Phase 1 is awaiting budget approval
  ([`PROPOSAL.md`](PROPOSAL.md) §7).
- **License/provenance:** derived from corpus v2; same constraints. No new
  generation, no new sources.
- **Related logs:** [`PROPOSAL.md`](PROPOSAL.md) §3,
  [`EXPERIMENTS.md`](EXPERIMENTS.md) §12.
