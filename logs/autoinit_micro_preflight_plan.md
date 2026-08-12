# Micro-preflight plan — measure five things, then delete the pod

**Status: PROPOSAL, NOT AUTHORIZED. Nothing has been launched.**

This is **not** Phase A. It is a short, single-GPU measurement session whose only
purpose is to replace five estimates with numbers, so the Phase A authorization is
based on measurement rather than on a range. **Phase A must not start
automatically when it finishes.**

---

## 1. Why it exists

Two Phase A figures are currently ranges or placeholders, and both are ranges for
the same reason — one unmeasured quantity:

* the activation-statistics pass is priced between "GPU forward only" and "the
  measured CPU end-to-end rate", a **3.8× spread** that propagates to the whole
  search cost;
* beam ε is justified against **CPU** repeatability, which was exactly 0.0, but
  the pilot's evaluation backend is a GPU where reduction order is not guaranteed
  across launches;
* the recovery feasibility floor has a frozen *rule* and no number, because the
  canonical control has never been measured on the new recovery-search battery.

Guessing any of these and discovering it on a running Phase A would cost more than
this session does.

## 2. What is measured

| # | measurement | why it cannot be done at $0 |
| --- | --- | --- |
| 1 | activation-statistics wall clock, GPU forward vs float64 CPU accumulation, at the real 4B teacher over the real 59,763-position mixture | needs the 4B teacher on a GPU; the dev box has no GPU and the teacher is not local |
| 2 | state-evaluator repeatability: the same materialized checkpoint scored 10× on the real suite, reporting the per-objective range | GPU reduction non-determinism cannot be observed on CPU |
| 3 | peak GPU resident memory during the widest operator (DEPTH on the full-width teacher: parent + child + statistics) | the 14.3 GiB figure is arithmetic, never measured |
| 4 | canonical control on `recovery_search_v1`: `usable_rollout_rate`, `correct_overall`, `correct_given_usable`, per-set breakdowns | needs rollouts from the retained 0.86M-recovered checkpoint |
| 5 | checkpoint write/read throughput for a 5.99 GiB depth-only intermediate on the pod's container disk | the 106 GiB working-set plan assumes disk is not the bottleneck |

Measurements 1–3 use the real teacher and the real state-eval suite. Measurement 4
uses the **retained** `e1_r0860k_*_pca` recovery checkpoint — no training runs in
this session.

## 3. What is explicitly **not** done

* no beam search, no operator search, no Phase A;
* no recovery training of any kind;
* no searched candidate is created, scored or ranked;
* no threshold is chosen after seeing a searched candidate — measurements 1–3 are
  properties of the machinery and 4 is a property of the incumbent.

## 4. Hardware and shape

| | |
| --- | --- |
| GPU | 1 × L40S 48 GB, $0.99/h — the same card E8a's throughput anchor was measured on |
| container disk | ≥ 60 GiB (teacher 7.5 + one depth-only intermediate 6.0 + margin) |
| image | the session's standard training image |
| expected wall clock | ~55 min including setup |

An A100 is not needed: the widest measured step is ~14 GiB of resident memory, and
using a different card would also break comparability with the 88.83 TFLOP/s anchor
this project already measured on an L40S.

## 5. Cost

| item | expected | hard |
| --- | ---: | ---: |
| setup, teacher stage-in, suite stage-in | $0.25 | $0.50 |
| (1) statistics-pass split, 3 repeats | $0.15 | $0.30 |
| (2) evaluator repeatability, 10 scorings | $0.20 | $0.40 |
| (3) peak-memory probe on the widest operator | $0.15 | $0.30 |
| (4) control on the recovery-search battery, 190 prompts | $0.30 | $0.60 |
| (5) disk throughput | $0.05 | $0.10 |
| setup/redraw reserve (setup has varied 30× on this project) | — | $1.80 |

**Expected $1.10. Maximum authorized: $4.00.**

The reserve is the honest part of this number: this project's pod setup has taken
5, 8.5 and 150+ minutes for the same script and image, and `--terminate-after` has
never been observed to fire. Teardown is a verification poll, not a request.

### Budget position

```
actual cumulative spend                $180.7033
authorized cumulative cap              $211.07
unused, uncommitted                     $30.3667
this micro-preflight, maximum            $4.00
remaining for Phase A afterwards        $26.3667
```

## 6. Stop rules

The session **stops and reports** rather than continuing if:

* setup exceeds 40 minutes — redraw once, then stop;
* measurement 2 shows a per-objective range **≥ 1e-4**, i.e. at or above the
  declared beam ε. This is a *finding*, not a failure: ε is reset from the
  measured range in the preregistration before Phase A, and the remaining
  measurements still complete;
* peak resident memory exceeds 40 GiB on an L40S — the storage and hardware plan
  is wrong and Phase A needs re-pricing before it is booked;
* spend reaches $4.00.

## 7. Outputs, and what they change

1. **Statistics-pass split** → collapses the Phase A search cost from
   $0.93–3.57 to a point estimate.
2. **Evaluator repeatability** → confirms or resets beam ε in the preregistration.
3. **Peak memory** → confirms or replaces the 14.3 GiB arithmetic figure, and with
   it the "L40S is sufficient" claim.
4. **Control behaviour** → fills the feasibility floor and the per-capability
   catastrophic floor via their already-frozen rules.
5. **Disk throughput** → confirms or corrects the 106 GiB / 135 GiB working-set
   plan.

Then: **delete the pod, verify it is gone, and stop for review.** The Phase A
authorization request is rewritten from these numbers and brought back
separately. The current `$17.00 expected / $26.21 hard` figures are informative
and are **not** the authorization being requested here.

## 8. Session contract

[`scripts/pod/AGENTS.md`](../scripts/pod/AGENTS.md) applies unchanged: detached
start via `start_job.py`, `watchdog.py` beside the launcher from creation,
`LogRelay` mirroring continuously, `collect_artifacts.py` gating teardown, and
termination confirmed by polling the control plane rather than by a return code.

Before launch, the suite runs the two ways the E8b failures did not:

```
OMP_NUM_THREADS=8 taskset -c 0-12 python -m pytest tests -q
HIDDEN_PATHS="..." OMP_NUM_THREADS=8 taskset -c 0-12 bash scripts/pod/simulate_pod_env.sh
```
