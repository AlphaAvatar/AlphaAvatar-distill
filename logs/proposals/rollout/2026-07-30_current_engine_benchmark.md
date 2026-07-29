# 2026-07-31 — Current-engine rollout benchmark (pre-registration)

**Status:** pre-registered, **not run, no pod created, no spend committed.**
Replaces [`2026-07-30_rollout_engine_comparison.md`](2026-07-30_rollout_engine_comparison.md),
which is **retired unrun** because it was built around vLLM 0.11.0.

## 1. What was wrong with the previous plan

It treated **vLLM 0.11.0** as "the first measured engine". That build was reached
by pinning *backwards* until the engine fit this project's training image, after
vLLM **0.26.0** — the current stable release — failed to start. The gap spans
scheduler, kernels, CUDA-graph behaviour, throughput, log-prob support and
operational characteristics. A compatibility build is not the engine.

The 0.26.0 failure was **environment selection, not an engine result**: the host
ran driver 570.124.06 / CUDA 12.8, and current vLLM targets CUDA 13, which needs
driver **≥580**. A venv cannot repair a host-driver mismatch.

It was avoidable with one flag. `runpodctl pod create` accepts
**`--min-cuda-version`**, and RunPod's `allowedCudaVersions` includes **13.0**.
The pod was created without it and landed on a CUDA-12.8 host by chance. **L40S
hosts with driver 580.95.05 / CUDA 13.0 exist**, so the GPU was never the
problem.

## 2. Objective

Choose the project's **rollout service** — one engine reused across Stage 3
corpus builds and Stage 4/5 rollouts — by comparing current production-grade
releases, each in its own engine-native environment, and by validating the
already-built rollout snapshot and importance-correction path against them.

**HF `model.generate` is not a candidate.** It appears only as the trainer-policy
scorer that supplies the numerator of the importance ratio, and consumes no
benchmark arm.

## 3. Arms

Each engine runs its **own official image**, so this is **one pod per engine**,
not one pod per session.

| arm | image | engine |
|---|---|---|
| **A — vLLM** | `vllm/vllm-openai:v0.26.0` | current stable |
| **B — SGLang** | `lmsysorg/sglang:v0.5.12` (unified NVIDIA tag) | current stable, deterministic mode where available |
| **C — substitute** | only if B cannot be run on available infrastructure | LMDeploy or TensorRT-LLM |

Arm C exists because **an infrastructure failure is not an engine loss**. If
SGLang cannot be scheduled on a compatible host, it is recorded as untested and
a third candidate takes its place, rather than SGLang being marked as losing.

## 4. Hardware and image selection — decided before launch

**Requirement:** CUDA 13 runtime with host driver ≥580, enforced at creation with
`--min-cuda-version 13.0`. This is the control whose absence caused the previous
failure and it is now mandatory for every arm.

Measured RunPod pricing (secure / community) and current stock:

| GPU | mem | secure | community | stock | note |
|---|---:|---:|---:|---|---|
| **L40S** | 48 GB | $0.99 | $0.79 | Low | Ada; mature support in both engines; CUDA-13 hosts confirmed to exist |
| RTX PRO 4500 | 32 GB | $0.74 | $0.34 | **Medium** | Blackwell; cheapest capable, best availability |
| RTX 5090 | 32 GB | $0.99 | $0.69 | Low | Blackwell consumer |
| H100 SXM | 80 GB | $2.99 | $2.69 | **Medium** | best-supported, ~3× the cost |

**Recommendation: L40S, `--min-cuda-version 13.0`**, on these grounds — 48 GB
removes memory as a confound (a 48 GB card already OOMed once in this project at
long context), Ada has the most mature kernel coverage in *both* engines, and
$0.99/h is mid-range. **Fallback order if allocation fails:** RTX PRO 4500 →
RTX 5090 → H100 SXM. A fallback changes GPU architecture, which is a confound, so
whichever host is used, **both arms run on the same GPU type** or the comparison
is void.

## 5. Models and job shapes

The rollout service serves two different workloads and both are measured:

* **Stage 3 shape — teacher `Qwen3-4B-Thinking-2507@768f209d`**, cap 4096,
  in-scope slices only (`rag_evidence, multihop_qa, gsm8k, openmath`). This is
  the expensive workload and it sizes the corpus.
* **Stage 4/5 shape — the student** (`stage3/s2v1_from_init/step_002700`), the
  policy that will actually be rolled out. Cheap, and it is the model the
  correction path must be validated on.

## 6. What is compared

Per the maintainer's list, and explicitly **not** exact cross-engine token
identity, which is retired as a gate and kept only as a diagnostic:

1. **Throughput and cost** — tok/s and $/1k prompts at both job shapes.
2. **Long-context behaviour** — cap sweep 4096 / 8192 / 16384, reporting
   completion rate and throughput decay, not just a single point.
3. **Token/log-prob transport** — token ids in and out with no text round-trip;
   per-token rollout log-probs present, correctly aligned, and correctly
   trimmed at the stop token.
4. **Batch sensitivity** — throughput and output stability across batch sizes,
   including whether batching changes emitted tokens (both engines are already
   known not to be batch-invariant; the question is how much).
5. **Operational reliability** — startup time, image pull time, restart
   behaviour, failure modes, and whether the server survives a malformed request.
6. **Corrected-training results** — the importance-ratio distribution against the
   trainer policy, and a small corrected-training pilot.

## 7. Environment identity, pinned and reported per arm

Recorded into every report, because a benchmark that cannot state these does not
support an adoption decision (P4):

engine version/commit · **official image digest** · CUDA runtime · **host driver
version** · torch · transformers · GPU architecture · dtype and quantization ·
attention and sampling backend · observed token/log-prob API behaviour.

The report fails loudly rather than emitting a row with any of these unknown.

## 8. The correction experiment

Unchanged in substance from the retired proposal — the machinery is already built
and CPU-tested (`aadistill.rollout`, 218 tests).

Generate student rollouts on each engine with log-probs, snapshot them hashed,
then score the trainer policy on those exact token ids and compute the ratio
distribution. **Trainer-policy scoring runs in the project environment, not the
engine image**, so the numerator comes from the real training stack; the student
is 0.6B, so this is cheap and can run on the dev box.

**Pre-registered stability bound**, unchanged and set before data: corrected
training stays within the band a same-seed reference pilot occupies; the
clipped/rejected token fraction stays **below 5%**; and the corrected run is not
worse than an uncorrected control on the guard rail.

## 9. Budget

| item | estimate |
|---|---|
| Arm A — vLLM pod (image pull, model download, benchmark, rollouts) | ≤ 1.5 h |
| Arm B — SGLang pod, same | ≤ 1.5 h |
| GPU rate (L40S secure) | $0.99/h |
| **ceiling** | **≤ 3.0 h ≈ $3.00**, plus ≤$1 contingency for a failed allocation |
| `--terminate-after` | +4 h absolute, per pod |

Arms run **sequentially**, so a failure in A is diagnosed before B is paid for.
No corpus is built and no training runs; the corrected-training pilot is a
separate, later request once an engine is chosen.

## 10. Abort rules

* **A1.** If a pod cannot be allocated with `--min-cuda-version 13.0` after the
  fallback order in §4, stop and report. Do **not** relax the CUDA constraint —
  that is precisely the error being corrected.
* **A2.** If an engine's official image fails to start, record the failure with
  full environment identity and move to the next arm. Do not downgrade the
  engine to make it run.
* **A3.** If an engine cannot supply per-token rollout log-probs, record it and
  finish the arm's throughput measurements anyway; log-prob support is an
  adoption criterion, not a reason to discard throughput data.
* **A4.** Hard stop at the 3.0 h ceiling.

## 11. What this proposal does not do

* It does not carry vLLM 0.11.0's numbers forward. They are measurements of an
  obsolete compatibility build on a mismatched host and are not "vLLM's".
* It does not adopt any engine. Adoption needs this benchmark **plus** the
  corrected-training pilot.
* It does not build a corpus.
