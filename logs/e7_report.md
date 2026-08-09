# Experiment 7 — FineWeb teacher-KD mixture at the fixed 1.60M rung

Generated 2026-08-09T23:20:41.416065+00:00 from retained generations only. Inclusion
mask `d6e24e0b09da1bcc…`, 150 prompts, greedy,
unrestricted generation (P18), every arm re-scored with the current
scorer. Arm A is the **retained** E1/P1 KD-heavy 1.60M baseline — the
same generations E6 produced, not a new run.

All three arms train the identical 1.60M rollout stream. B and C differ
from A by an added KD-only stream and from each other **only** in that
stream's content: both consume exactly 1,801,503 extra KD positions.

## 1. General-language restoration — DIAGNOSTIC ONLY

These may not promote a checkpoint (decision record 2026-08-09). They
answer whether general language modelling came back, and nothing else.

| arm | FineWeb NLL | teacher KL | top-1 | mean rank |
| --- | ---: | ---: | ---: | ---: |
| E7-A-Baseline-sa | 9.4847 | 7.3504 | 0.0319 | 10177.6 |
| E7-A-Baseline-sb | 9.4541 | 7.3200 | 0.0334 | 7854.4 |
| E7-B-FineWeb-sa | 4.2664 | 1.9445 | 0.2845 | 511.4 |
| E7-B-FineWeb-sb | 4.2478 | 1.9307 | 0.2848 | 502.3 |
| E7-C-Control-sa | 4.7713 | 2.4555 | 0.2423 | 709.7 |
| E7-C-Control-sb | 4.7508 | 2.4443 | 0.2429 | 696.9 |

## 2. Autonomous behaviour — THE PROMOTION CRITERION

| arm | seed | usable | correct | correct\|usable | nat. term | ctx limit | repetition | empty |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A-Baseline | sa | 0.7800 | 0.1733 | 0.2222 | 0.8133 | 0.1867 | 0.1867 | 0.1400 |
| A-Baseline | sb | 0.6800 | 0.2000 | 0.2843 | 0.7067 | 0.2933 | 0.3067 | 0.2200 |
| **A-Baseline** | **mean** | **0.7300** | **0.1867** | 0.2511 | 0.7600 | 0.2400 | 0.2467 | 0.1800 |
| B-FineWeb | sa | 0.7467 | 0.2133 | 0.2857 | 0.7800 | 0.2200 | 0.2400 | 0.2000 |
| B-FineWeb | sb | 0.7133 | 0.1667 | 0.2336 | 0.7333 | 0.2667 | 0.2733 | 0.2067 |
| **B-FineWeb** | **mean** | **0.7300** | **0.1900** | 0.2603 | 0.7567 | 0.2434 | 0.2566 | 0.2034 |
| C-Control | sa | 0.8133 | 0.1533 | 0.1885 | 0.8200 | 0.1800 | 0.1867 | 0.1533 |
| C-Control | sb | 0.6867 | 0.1467 | 0.2136 | 0.7133 | 0.2867 | 0.3133 | 0.2400 |
| **C-Control** | **mean** | **0.7500** | **0.1500** | 0.2000 | 0.7667 | 0.2334 | 0.2500 | 0.1966 |

`usable_rollout` is reported with every component rate, never as a
weighted average. It is blind to correctness by construction, and its
components are not independent — `protocol_valid` subsumes two of them.

## 3. Paired comparisons on the shared mask

### B vs A — FineWeb + extra KD, total effect
* **usable** pooled Δ +0.0000 (floor 0.0800) — inside the floor — a tie; seeds {'sa': -0.0333, 'sb': 0.0333}, seed-consistent False
  * sa: 0.7800 → 0.7467, win/tie/loss 16/113/21, 95% CI [-0.1133, +0.0467]
  * sb: 0.6800 → 0.7133, win/tie/loss 23/109/18, 95% CI [-0.0533, +0.1200]
* **correct** pooled Δ +0.0033 (floor 0.0600) — inside the floor — a tie; seeds {'sa': 0.04, 'sb': -0.0333}, seed-consistent False
  * sa: 0.1733 → 0.2133, win/tie/loss 19/118/13, 95% CI [-0.0333, +0.1133]
  * sb: 0.2000 → 0.1667, win/tie/loss 7/131/12, 95% CI [-0.0933, +0.0200]

### C vs A — matched extra KD alone
* **usable** pooled Δ +0.0200 (floor 0.0800) — inside the floor — a tie; seeds {'sa': 0.0333, 'sb': 0.0067}, seed-consistent True
  * sa: 0.7800 → 0.8133, win/tie/loss 14/127/9, 95% CI [-0.0267, +0.1000]
  * sb: 0.6800 → 0.6867, win/tie/loss 30/91/29, 95% CI [-0.0933, +0.1067]
* **correct** pooled Δ -0.0367 (floor 0.0600) — inside the floor — a tie; seeds {'sa': -0.02, 'sb': -0.0533}, seed-consistent True
  * sa: 0.1733 → 0.1533, win/tie/loss 12/123/15, 95% CI [-0.0867, +0.0467]
  * sb: 0.2000 → 0.1467, win/tie/loss 8/126/16, 95% CI [-0.1200, +0.0067]

### B vs C — FineWeb content, beyond extra KD
* **usable** pooled Δ -0.0200 (floor 0.0800) — inside the floor — a tie; seeds {'sa': -0.0667, 'sb': 0.0267}, seed-consistent False
  * sa: 0.8133 → 0.7467, win/tie/loss 14/112/24, 95% CI [-0.1467, +0.0133]
  * sb: 0.6867 → 0.7133, win/tie/loss 27/100/23, 95% CI [-0.0667, +0.1200]
* **correct** pooled Δ +0.0400 (floor 0.0600) — inside the floor — a tie; seeds {'sa': 0.06, 'sb': 0.02}, seed-consistent True
  * sa: 0.1533 → 0.2133, win/tie/loss 16/127/7, 95% CI [+0.0000, +0.1200]
  * sb: 0.1467 → 0.1667, win/tie/loss 12/129/9, 95% CI [-0.0400, +0.0800]

## 4. The three questions, kept separate

The preregistration fixed this reading before the run (`e7_preregistration.md` §7.4):

1. **general-language restoration** — section 1, diagnostics only;
2. **autonomous stability** — `usable_rollout` and its components;
3. **autonomous reasoning correctness** — `correct_overall`, `correct_given_usable`, GSM8K.

**If B improves the general-text diagnostics but does not beat C on
autonomous correctness, FineWeb did not solve the reasoning
bottleneck.** A restored NLL is not a restored capability, and this
report must not be summarised as though it were.

