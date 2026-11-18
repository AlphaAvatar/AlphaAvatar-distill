# Teacher-parent initialization vs the packed Top-1 (TPI1): report

Stage 1 owns this round; the training execution stage is `stage3_recovery`. The design and how it was run are in
[`README.md`](README.md); the evidence is in
[`logs/stages/stage-1/teacher_parent_initialization/`](../../../../logs/stages/stage-1/teacher_parent_initialization/).
Recorded 2026-11-18 (UTC). Terminal state: **COMPLETE** (every registered unit measured, the primary comparison
formal).

## 1. Result

**The registered primary comparison does not distinguish the two arms.** In thinking mode, over the static
fluid-reasoning formal core (570 items in 558 independence groups, both training seeds pooled, 1,140 paired records
per arm):

| | teacher-parent init (A) | packed Top-1 (B) | A − B | test |
| --- | --- | --- | --- | --- |
| thinking mode (primary) | 11.67% (133 / 1,140) | 10.26% (117 / 1,140) | **+1.40 pp**, 95% interval [−0.53, +3.45] | p = 0.195: **NOT_DISTINGUISHED** |
| non-thinking mode (key secondary) | 13.60% (155 / 1,140) | 11.23% (128 / 1,140) | +2.37 pp, interval [+0.61, +4.23] | not tested (fixed sequence): descriptive |

* The test is the registered group-level paired sign-flip test (100,000 draws, `default_rng(20261117)`): 93 of the
  558 groups differ, |Σ d| = 16, 19,542 draws at least as extreme. The interval is the group bootstrap
  (10,000 resamples, `default_rng(20261118)`), shown beside the test and not the decision.
* NOT_DISTINGUISHED is not equivalence. The interval allows A to be about half a point worse or about three and a half
  points better.
* The non-thinking interval excludes zero, but it is unadjusted, and under the registered fixed sequence the key
  secondary is tested only if the primary rejects. It did not, so the non-thinking difference is a description and not
  a finding.
* Both seeds agree in sign in both modes (per-seed table in section 4). Two training seeds bound what can be said about
  training randomness; the test's uncertainty is the items'.
* eval-set-1 is development-exposed: nothing here is a never-seen blind result.

The formal output is [`analyses/teacher_parent.json`](../../../../logs/stages/stage-1/teacher_parent_initialization/analyses/teacher_parent.json)
(`analysis.py`, `status: formal`; every A record's protocol equals B's, no unit incomplete).

Any difference between the arms belongs to the whole A recipe: every selection made on the teacher, and the residual
PCA re-aggregated from the teacher's statistics at the kept layers. It is never attributable to one of those changes.

## 2. What was compared

The two arms have the same 596M structure (28 layers, width 1024, FFN 3072, 16 / 8 heads, tied embedding, 596,049,920
parameters), the same operators and profiles, the same teacher (`Qwen/Qwen3-4B@1cfa9a72`, the hybrid) and the same
Stage-3 recipe (A-MIX-6cat-sr: 9,701 optimizer steps, 3 epochs of 6,467 packed sequences of 8,192 tokens, CE 0.25 +
Top-200 + tail KD 1.0). Only the initialization differs ([`README.md`](README.md) section 1):

* **B** (`B_packed_top1`, the control): the packed search's Top-1, each operator selected on the previous operator's
  child. Its sealed A-MIX evaluations of both seeds were reused, never retrained
  ([`audit/b_evidence.json`](../../../../logs/stages/stage-1/teacher_parent_initialization/audit/b_evidence.json)).
* **A** (`A_teacher_parent`): FFN, DEPTH and ATTENTION selections all made on the unmodified teacher (each rebuilt on
  the teacher and equal to its journal digest bit for bit), and the width PCA taken from the teacher's own residual
  statistics at the 29 points `sorted({0, *K}) + [36]` with weights 9 / 1 / 8 (captured energy 0.9256), built from one
  frozen mapping in the order FFN → DEPTH → WIDTH → ATTENTION. Init session `tpi_init_20261117_142049`: artifact
  `0474006c…`, weights `9120eb2e…`.

## 3. Training

| | seed 20260726 | seed 20260801 |
| --- | --- | --- |
| A, validation CE / KD at step 0 | 22.085 / 21.941 | 22.085 / 21.941 |
| B, step 0 | 7.321 / 7.168 | 7.321 / 7.168 |
| A, step 1,212 | 1.062 / 0.917 | 1.049 / 0.913 |
| B, step 1,212 | 1.104 / 0.959 | 1.092 / 0.953 |
| A, step 2,424 | 0.892 / 0.749 | 0.897 / 0.758 |
| B, step 2,424 | 0.934 / 0.793 | 0.930 / 0.790 |
| A, step 9,701 (final) | **0.721 / 0.585** | **0.717 / 0.580** |
| B, step 9,701 (final) | 0.749 / 0.613 | 0.746 / 0.608 |
| A, median seconds per optimizer step | 3.27 | 3.29 |
| A, peak GPU memory allocated | 17.47 GB | 17.47 GB |

Validation is the recipe's fixed 64 packed sequences, teacher-forced (a diagnostic, not a capability measure).

* A starts far worse. Its step-0 validation CE is 22.09 against B's 7.32, and the init session's state evaluation
  (diagnostic, against the root teacher) gives an equal-domain mean KL of 19.58 against B's journal leaf's 6.78, with
  top-1 agreement 0.077.
* By step 1,212 (the first validation after the 485 warmup steps) A is below B, and it stays below at every later
  validation in both seeds. It ends 0.028 / 0.029 lower in CE and 0.028 / 0.029 lower in KD.
* AGENTS.md 4.2 warns that initialization NLL is not recovered behaviour. Here the step-0 ranking reversed after
  training: the initialization that is much worse at step 0 trains to a lower validation loss and is not worse on
  eval-set-1.

Every unit has one log across resumes (`evidence/A_teacher_parent-s<seed>/train/train_log.jsonl`). Seed 20260726's
log holds, before its `resume_loaded` event, the lost session's steps 2,209–2,540 (a discarded branch, never pooled;
section 6). The resumed branch repeats those steps' training loss to about 1e-3 (2210: 0.9374 vs 0.9363), and its step
2,424 validation CE is 0.8921 against the lost branch's 0.8960. That is not bit-exact, which is expected from
nondeterministic CUDA kernels.

## 4. Eval-set-1

Each seed was evaluated on every item of every unit, in both modes, under B's protocol exactly: `nucleus_v1`, the
trained context 8,192, action protocol v2, the hybrid's tokenizer files. Every A unit's item count equals B's, and
every fluid record's protocol equals B's (`protocol_identity`).

### Primary and key secondary, per seed

| mode | seed | A | B | A − B | 95% interval (per seed) |
| --- | --- | --- | --- | --- | --- |
| thinking | 20260726 | 11.75% | 10.35% | +1.40 pp | [−1.39, +4.24] |
| thinking | 20260801 | 11.58% | 10.18% | +1.40 pp | [−1.21, +4.00] |
| non-thinking | 20260726 | 12.81% | 10.70% | +2.11 pp | [−0.35, +4.59] |
| non-thinking | 20260801 | 14.39% | 11.75% | +2.63 pp | [+0.18, +5.05] |

B's thinking-mode 10.35% / 10.18% reproduce A-MIX's 10.26% (B's own audit).

### By reasoning axis (static formal core, both seeds pooled; descriptive)

| axis | n | A thinking | B thinking | A non-thinking | B non-thinking |
| --- | --- | --- | --- | --- | --- |
| task-contained deduction | 378 | 23.54% (89) | 21.69% (82) | 26.72% (101) | 24.87% (94) |
| inductive rule discovery | 284 | 3.52% (10) | 1.76% (5) | 5.28% (15) | 2.11% (6) |
| compositional generalization | 288 | 11.81% (34) | 9.72% (28) | 12.85% (37) | 8.68% (25) |
| abstract / analogical (sub-category) | 190 | 0.00% (0) | 1.05% (2) | 1.05% (2) | 1.58% (3) |

The counts per axis are small, especially for inductive rule discovery and abstract / analogical, so no axis is read
on its own.

### Interactive adaptation and planning (144 episodes per seed, formal core; descriptive)

| mode | A success | B success | A − B (pooled) |
| --- | --- | --- | --- |
| thinking | 2 / 288 (0.69%) | 2 / 288 (0.69%) | 0.00 pp, interval [−1.39, +1.39] |
| non-thinking | 17 / 288 (5.90%) | 13 / 288 (4.51%) | +1.39 pp, interval [−1.74, +4.51] |

In thinking mode most episodes of both arms end in a degeneration stop (A 220, B 250 of 288). A hits the context limit
more often (40 vs 10).

### Terminations apart from correctness (static core, both seeds, 1,140 records per arm)

| | A thinking | B thinking | A non-thinking | B non-thinking |
| --- | --- | --- | --- | --- |
| natural termination (EOS) | 805 | 754 | 1,003 | 992 |
| repetition / degeneration stop | 309 | 377 | 137 | 148 |
| context-limit hit (8,192; right-censored length) | 26 | 9 | 0 | 0 |
| usable rollout | 804 (70.5%) | 751 (65.9%) | 1,003 (88.0%) | 992 (87.0%) |
| mean generated tokens, seed 1 / seed 2 | 2,922 / 3,467 | 2,882 / 2,911 | 696 / 741 | 623 / 710 |

A has fewer degeneration stops in thinking mode and more context-limit hits. Its seed-20260801 thinking-mode
generations are longer on average. A context-limit hit stays in every denominator and is scored by the unit's scorer
like any other record.

### Supporting units (registered headline, per seed: A / B; descriptive)

| unit | n | thinking s1 | thinking s2 | non-thinking s1 | non-thinking s2 |
| --- | --- | --- | --- | --- | --- |
| fluid_reasoning_720 (whole unit, its own headline) | 714 | 9.66 / 8.54 | 9.24 / 8.12 | 11.48 / 9.52 | 12.61 / 10.22 |
| gsm8k (math, a learned procedure) | 100 | 8 / 5 | 13 / 11 | 10 / 4 | 5 / 4 |
| math_verified | 100 | 1 / 0 | 1 / 0 | 2 / 2 | 0 / 0 |
| multihop (reading from supplied evidence) | 100 | 35 / 28 | 34 / 30 | 51 / 52 | 44 / 51 |
| rag (reading from supplied evidence) | 100 | 47 / 49 | 45 / 43 | 56 / 57 | 61 / 61 |
| knowledge (closed-book) | 150 | 2.0 / 2.0 | 2.0 / 2.67 | 3.33 / 2.67 | 3.33 / 3.33 |
| answerability_paired (pair correct) | 120 | 1.67 / 1.67 | 5.0 / 1.67 | 6.67 / 5.0 | 6.67 / 5.0 |
| safety_paired (pair correct) | 100 | 0 / 0 | 0 / 0 | 4 / 2 | 2 / 2 |
| stage3_legacy_150 (in-sample regression; never generalization) | 150 | 41.33 / 40.0 | 44.67 / 38.67 | 47.33 / 47.33 | 48.0 / 46.67 |

All values are percentages. `eval_behavior_v0` (65) has no registered correctness headline.

## 5. What the results say, and what they do not

* **They say:** under the fixed structure and the A-MIX recipe, initializing from teacher-level selections with
  teacher statistics does not measurably change thinking-mode static fluid accuracy against the packed Top-1. The
  estimate is +1.40 pp in each seed. It does not reach the registered threshold, and the plausible range runs from
  about −0.5 to +3.5 pp.
* **They say:** a much worse initialization by teacher-forced measures (step-0 CE 22 vs 7, state KL 19.6 vs 6.8) is
  not a worse student after this recovery. On validation CE and KD it is slightly better from step 1,212 on, in both
  seeds.
* **They do not say:** that the arms are equivalent; that the non-thinking +2.37 pp is real (untested under the
  registered sequence); that either change inside A is responsible; that the result holds for another recipe, a longer
  recovery or a held-out blind set. Absolute fluid accuracy for both arms is low (10–14%).
* **For a later round (a planning note, not a claim):** the interval's half-width is about 2 pp on 1,140 paired
  records. Detecting an effect of about 1.4 pp at conventional power would need several times as many independent
  items, not more training seeds, because the test's uncertainty is the items'.

## 6. Execution

| session | what | outcome | all-in |
| --- | --- | --- | --- |
| `tpi_init_20261117_142049` | build A, gates, statistics, state evaluation, relay copies | COMPLETE, 22.8 min | $0.4217 |
| `tpi_train_20261117_145734` | training | aborted at $0 before any create (the cap at the $1.11/h ceiling exceeded $28) | $0.0000 |
| `tpi_train_20261117_145939` | training, seed 20260726 to step 2,540 | lost to the provider (DRIVER_UNKNOWN), 166.3 min ([`failure.json`](../../../../logs/stages/stage-1/teacher_parent_initialization/runs/tpi_train_20261117_145939/evidence/failure.json)) | $3.0779 |
| `tpi_train_20261117_180111` | resume seed 20260726 at the verified `step_002208`; seed 20260801; all four evaluations | COMPLETE (ALL_DONE), 1,060.7 min | $19.6325 |

All runs were on one L40S at $1.09/h, secure cloud, with no network volume. Times are UTC.

* **The replacement session** (pod `tdxvznkx54lc4q`, 2026-11-17 18:03Z to 2026-11-18 11:44Z):
  * It restored the save point from the relay: weights, AdamW state, step, consumed sequences, and CPU and CUDA RNG.
  * Seed 20260726: training from 18:10Z to 01:15Z (7,493 steps, 25,505 s); evaluations 01:16–01:44Z (thinking) and
    01:44–01:50Z (non-thinking).
  * Seed 20260801: training from 01:50Z to 11:01Z (9,701 steps, 33,054 s; B's were 32,322 / 32,394 s); evaluations
    11:02–11:35Z and 11:35–11:41Z.
  * A's evaluations took 28–33 min (thinking) and 6–7 min (non-thinking), against the 67 min / 19–23 min the
    proposal priced from B's. The difference is not analysed here.
  * The launcher deleted the pod and the provider confirms it gone. The account holds no pod and no volume, the
    watchdog ended `pod_gone`, and no observer of the run is left.
* **Save points.**
  * Every 1,104 steps, each save point (weights + AdamW state + RNG + data position, about 5.9 GB) went to the relay
    and was verified by a fresh download and re-hash.
  * The previous save point was permanently deleted only after the next one verified. All save points were deleted
    once each final verified.
  * Uploads took 37–49 s.
* **Relay slowdown (observation).**
  * From about 06:00Z on 11-18 the verify download from the private relay slowed: seed 20260801's `step_004416` took
    1,294 s and `step_005520` 2,311 s, against about 16 s otherwise.
  * A read-only probe on the same pod measured about 4.4 MB/s from the private repository against 104 MB/s from the
    public HF CDN. The bottleneck is the hub's storage path for the private repository, not the pod.
  * It recovered by 08:07Z (`step_006624` took 16 s; `step_007728` 94 s, `step_008832` 47 s), and the finals
    verified in 7 s.
  * Training was unaffected. But the driver's evidence copy (`copy_small`) shares the mirror thread, so the copying
    of the training log to the dev box paused for up to 39 min during each slow verify, and the launcher log went
    silent. Evaluations run after training, when the mirror is idle, so their files were not delayed.
  * A driver that must survive a long relay stall would run the two in separate threads. This was not changed during
    the run (never edit a running script).
* **Repair after the run.** `relay.get` now moves the downloaded blob out of its temporary cache instead of copying
  it, so fetching a 2.2 GiB final needs one copy's space on the dev box, not two (8.1 GB was free).

## 7. Cost

| | USD |
| --- | --- |
| init session | 0.4217 |
| aborted training artifact | 0.0000 |
| lost training session | 3.0779 |
| replacement training session | 19.6325 |
| **TPI1 total** | **23.1321** |
| ceiling (sessions) / teardown reserve | 31.50 / 1.00 |
| unspent, lapses (funds nothing later) | 8.3679 |

The proposal expected $25.20. Each all-in is GPU at the accepted price plus 150 GB of container disk at
$0.1/GB-month, from the run's `closeout/outcome.json`. The project cap was not raised.

## 8. Checkpoints and storage

| artifact | sha256 (weights) | kept |
| --- | --- | --- |
| A's initialization (bf16) | `9120eb2e…` | dev box `~/aad-artifacts/stage1_teacher_parent_init/init/init/01e63327e3d02b2f/` |
| the frozen mapping | `aae2e9b8…` (tensors) | dev box and relay |
| the teacher's residual statistics | `e4145e38…` | dev box and relay |
| A, seed 20260726 final (fp32, 2,384,234,968 B) | `1a02ba42…` | dev box `~/aad-artifacts/stage1_teacher_parent_init/A_teacher_parent-s20260726/final/1a02ba42945d8a82/` |
| A, seed 20260801 final (fp32, 2,384,234,968 B) | `19396052…` | dev box `~/aad-artifacts/stage1_teacher_parent_init/A_teacher_parent-s20260801/final/193960520c006de5/` |
| the session's evidence (generations, rows, reports, logs; no weights) | `f9cdf22c…` (tgz) | dev box (archive + extracted) and relay |

The dev-box record is
[`storage/local_retention.json`](../../../../logs/stages/stage-1/teacher_parent_initialization/storage/local_retention.json);
each file there was re-hashed against the relay record the pod verified. After that verification the relay copies of
the three weight files (the init and both finals) were retired under AGENTS.md 2.5, and each has a tombstone in
`logs/maintenance/inventories/checkpoint_tombstones.json`. No consumer remains: the study is closed, so no session can be
issued that would restore from the relay. The mapping, the statistics, the configs and the evidence archive stay on
the relay. No save point remains anywhere.
After the retention the dev box had about 3.5 GB free on `/`, a disk it shares with another project, and it now holds
the only copy of A's weights.

## 9. Reproducing

```
PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \
  scripts/stages/stage-1/teacher_parent_initialization/analysis.py \
  --a-root ~/aad-artifacts/stage1_teacher_parent_init/tpi_train_20261117_180111/extracted
```

This reproduces `analyses/teacher_parent.json` from the archived A generations and B's
(`~/aad-artifacts/stage3_sr/a_train_sr_mix_20261109_193709/extracted/`). The archive's sha256 and the files committed
from it are in `runs/tpi_train_20261117_180111/evidence/evidence_archive.json`. The executables are the session
commit `e8dc921d` (the bundle `aad_autoinit_e8dc921d.bundle`), and the commands are in [`README.md`](README.md)
section 5.
