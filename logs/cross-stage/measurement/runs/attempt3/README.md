# Bounded measurement, attempt 3 — 2026-08-20, $0.2077, **COMPLETE**

**The measurement ran.** `ALL_DONE`, both fail-closed conditions passed, artifact
manifest `rc=0`, teardown on the normal gate, pod deleted with provider
confirmation. **Authorization CONSUMED — it covered one launch and must not be
reused.**

| | |
| --- | --- |
| authorization | `autoinit.measurement.2026-08-20T0512Z`, sha256 `13ebf61a36a3fea2ebc583ec0f4624ecb891155e18c0cf5c453ad51e20cfe1d8` |
| grant | [`../autoinit_measurement_grant3.json`](../autoinit_measurement_grant3.json), sha256 `2124eaef0fc5dcd1658db72971696b8da878e2ba65e854e15925bfec2860d72e` |
| authorized base | `e6a0cef07f332d737fb535948bd307abebd4d2b5` |
| session commit | `88349d70e6413460164818fbd408b91989d3aa75` (authorization-only, one path) |
| harness digest | `2838019de3e9750de68f420716358261453773e28e7a2cb222624f1ec49c053b` (13 files) |
| bundle | `aad_autoinit_88349d70.bundle`, sha256 `ca879159ce149bb21bd1f18050c1be92c3397eef85d4e24f4ecc1bd25c89830c` |
| pod | `fsk7tz1rnx43xr`, 1×L40S @ $0.99/h, first draw, stock High |
| lifetime | 05:35:07 → 05:47:42 UTC, **12.59 min** |
| cost | **$0.2077** of a $1.6294 ceiling (plan's own hard stop $0.8910) |
| terminal | `ALL_DONE`, `passed: true`, no launcher error |

## The headline: the port matches E8a's throughput

| | |
| --- | --- |
| weighted 260-evaluation schedule | **21.53 min** |
| **weighted evaluations/minute** | **12.07** |
| E8a frozen cost model | 1,300 s = 21.7 min, **12.0/min** |
| attempt 10, same operator on the host | ≥ 647 min for **one** expansion, unfinished |

**12.07 against an anchor of 12.0 — 0.6% above.** The frozen cost model was
right, the repaired port reaches it, and the $11.43 attempt-10 failure is
confirmed as a placement bug rather than an intrinsic cost. The host path was at
least **30× slower** and never finished.

## All 24 timings, by skip cardinality

| \|skip\| | sample 0 | sample 1 | sample 2 | mean | schedule weight |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.27 | 5.24 | 5.24 | **5.251** | 36 |
| 2 | 5.16 | 5.18 | 5.18 | **5.175** | 35 |
| 3 | 5.10 | 5.11 | 5.11 | **5.105** | 34 |
| 4 | 5.00 | 5.00 | 4.99 | **5.000** | 33 |
| 5 | 4.90 | 4.92 | 4.91 | **4.910** | 32 |
| 6 | 4.82 | 4.85 | 4.82 | **4.829** | 31 |
| 7 | 4.72 | 4.75 | 4.73 | **4.734** | 30 |
| 8 | 4.62 | 4.64 | 4.64 | **4.635** | 29 |

Seconds. Weights sum to 260. Within-cardinality spread is ≤ 0.03 s, so the three
samples are measuring the same thing and the means are not hiding variance.

**The cost falls monotonically with cardinality**, 5.251 → 4.635, because
skipping more layers is less work per forward. That is exactly why the flat
extrapolation is wrong:

* weighted, over the real 36,35,…,29 schedule: **21.53 min**
* flat at cardinality 8 (**incorrect**, labelled as such in the artifact): **20.08 min**
* the flat figure **understates by 6.7%**

The schedule spends most of its evaluations at *low* cardinality, where each one
costs more. Pricing from the cheapest configuration is the mistake the label
exists to prevent.

## Backend equality: exactly zero

| pair | \|skip\| | max per-item KL delta | mean per-item delta |
| --- | ---: | ---: | ---: |
| `{0}` | 1 | **0.0** | **0.0** |
| `{0,5,10,15,20,25,30,35}` | 8 | **0.0** | **0.0** |

Not "below a threshold" — **identically zero** at both ends of the schedule, on
one accelerator, per item. The repaired port computes what E8a computes.

The aggregated scores differ by 0.0228 and 0.0340, and that is the **declared**
aggregation difference, not drift: E8a merges raw sums per subtype and normalizes
once (position-weighted); the operator normalizes per item and takes an unweighted
mean. Predicted ~0.027 from the CPU equivalence work, observed 0.023–0.034. This
is why the contract makes the **per-item** delta the comparison — the aggregate
difference is ~300× the 8.195e-05 decision margin and would be unreadable as a
backend check.

## VRAM, and the cache decision

| | |
| --- | --- |
| **production peak** (the Phase-A number) | **26.82 GiB** |
| comparison-path peak | 10.45 GiB |
| device total | 44.39 GiB |

The dual-cache repair holds: the production reference cache is released before
E8a builds its path, so neither peak carries two ~16.9 GiB caches. 26.82 GiB on a
44.39 GiB L40S leaves **17.6 GiB** of headroom.

```
cached            : true
estimate          : 16.913 GiB (18,160,302,336 B)
available         : 36.42 GiB (39,107,035,136 B)
budget fraction   : 0.66
headroom source   : cuda.mem_get_info:cuda
fallback          : null
```

The production `_ReferenceLogits` policy **caches** at the frozen mixture. The
priced basis stands; `MEASUREMENT_FALLBACK` was not reached.

## GPU utilization: the attempt-10 diagnosis, confirmed

| | attempt 3 | attempt 10 |
| --- | ---: | --- |
| mean | **98.3%** | 0–1% |
| median | 98% | — |
| min / max | 94% / 100% | — |
| fraction of samples below 10% | **0.0** | ~all, for 11 hours |

221 samples via `nvidia-smi`. **Not one sample below 10%.** Attempt 10's idle
accelerator was the `.cpu()` in `depth.causal_kl_greedy_v1`, and moving the
reduction back to the device is what fixed it.

The cgroup fix is also confirmed on hardware: `visible_cpus: 128`,
`torch_threads_before: 128`, `torch_threads_after: 13`, source `cgroup.v2`. The
container reported 128 CPUs it did not have; the driver held torch to the 13 it
was actually granted.

## Identities

| | |
| --- | --- |
| teacher | `Qwen/Qwen3-4B-Thinking-2507` @ `768f209d9ea81521153ed38c47d515654e938aea`, **pinned: true** |
| calibration | `calib.domain_balanced@v1`, profile hash `11f36a88bad4879f…`, **67 items, 59,763 positions** |
| vocab / layers | 151,936 / 36 |
| GPU | NVIDIA L40S, 44.39 GiB, driver `580.159.03` |
| image | `runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404` |
| runtime | Python 3.12.3, **torch 2.11.0+cu128**, **transformers 5.13.1**, cuda-toolkit 12.8.1 |
| teacher load | 3.03 s (weights already on disk from `TEACHER_READY`) |
| reference pass | 4.31 s |

`calibration_path` reads `/workspace/aad/artifacts/stage1/e8_calibration_v1/items.jsonl`
— 61 characters. Before this session's `$0` closure it would have been 734,042
characters of serialized mixture.

## The chain, and what it cost

Eight steps, every one verified before a provider was contacted: one-use grant →
clean pre-authorization base `e6a0cef0` → authorization → authorization-only
commit differing in **exactly one path** → bundle upload → round-trip verification
of bytes, checkout and harness digest **recomputed from the relay checkout** →
`$0` gates (suite 1958 passed/11 skipped; simulator 1918/22 with the artifact tree
restored exactly; frozen verifier passed and not weakened; plan/harness/stage
gates pass; **stage 1 refused**; `allows_phase_a` False) → one launch.

Setup passed on the **first draw** in 6.4 min. The driver started at 05:43:50 and
was done at 05:46:15 — **2 min 25 s** for load, 24 timed evaluations, the cache
decision and both E8a pairs.

Three attempts, **$0.4611** total. Attempt 1 ($0.0700) bought the setup contract;
attempt 2 ($0.1834) bought the entrypoint seam — and the seam then found two more
defects at $0 before this run. Attempt 3 spent $0.2077 and answered every question
it was authorized to ask.

## Disposition

Recorded as **consumed and COMPLETE**. The grant and the authorization are spent
and must not be reused. Cumulative spend $206.2664 → **$206.4741** of $219.00,
leaving **$12.5259**.

**These values are inputs to a repricing and a separate cumulative-budget
decision. They authorize nothing.** No Phase-A attempt 11 is prepared, granted,
funded or implied, and none is being prepared.
