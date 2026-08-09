# Experiment 6b — objective × data-scale interaction

Generated 2026-08-09T00:58:16.867800+00:00 from retained generations only. Inclusion
mask `d6e24e0b09da1bcc…`, 150 prompts, greedy,
unrestricted generation (P18), every arm re-scored with the current scorer.

## Headline

| model | unique CE | cumulative CE | seed | usable | correct | correct\|usable |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| E1-1.60M | 1,600,353 | 4,801,059 | sa | 0.7800 | 0.1733 | 0.2222 |
| E1-1.60M | 1,600,353 | 4,801,059 | sb | 0.6800 | 0.2000 | 0.2843 |
| **E1-1.60M** | | | **mean** | **0.7300** | **0.1867** | 0.2511 |
| E1-2.96M | 2,960,507 | 8,881,521 | sa | 0.8533 | 0.2133 | 0.2500 |
| E1-2.96M | 2,960,507 | 8,881,521 | sb | 0.8267 | 0.2000 | 0.2419 |
| **E1-2.96M** | | | **mean** | **0.8400** | **0.2067** | 0.2460 |
| P2-1.60M | 1,600,353 | 4,801,059 | sa | 0.7133 | 0.2333 | 0.3178 |
| P2-1.60M | 1,600,353 | 4,801,059 | sb | 0.7533 | 0.1667 | 0.2212 |
| **P2-1.60M** | | | **mean** | **0.7333** | **0.2000** | 0.2682 |
| P2-2.96M | 2,960,507 | 8,881,521 | sa | 0.7733 | 0.1800 | 0.2241 |
| P2-2.96M | 2,960,507 | 8,881,521 | sb | 0.7467 | 0.2000 | 0.2679 |
| **P2-2.96M** | | | **mean** | **0.7600** | **0.1900** | 0.2456 |

## Paired comparisons on the shared mask

### A P2-2.96M vs E1-2.96M
* **usable** pooled Δ -0.0800 (floor 0.0800) — above the floor and seed-consistent (worse); seeds {'sa': -0.08, 'sb': -0.08}, seed-consistent True
  * sa: 0.8533 → 0.7733, win/tie/loss 10/118/22, 95% CI [-0.1533, -0.0067] excludes 0
  * sb: 0.8267 → 0.7467, win/tie/loss 11/116/23, 95% CI [-0.1533, -0.0067] excludes 0
* **correct** pooled Δ -0.0167 (floor 0.0600) — inside the floor — a tie; seeds {'sa': -0.0333, 'sb': 0.0}, seed-consistent False
  * sa: 0.2133 → 0.1800, win/tie/loss 10/125/15, 95% CI [-0.1000, +0.0333]
  * sb: 0.2000 → 0.2000, win/tie/loss 10/130/10, 95% CI [-0.0600, +0.0600]

### B P2-2.96M vs P2-1.60M
* **usable** pooled Δ +0.0267 (floor 0.0800) — inside the floor — a tie; seeds {'sa': 0.06, 'sb': -0.0067}, seed-consistent False
  * sa: 0.7133 → 0.7733, win/tie/loss 25/109/16, 95% CI [-0.0200, +0.1467]
  * sb: 0.7533 → 0.7467, win/tie/loss 21/107/22, 95% CI [-0.0933, +0.0800]
* **correct** pooled Δ -0.0100 (floor 0.0600) — inside the floor — a tie; seeds {'sa': -0.0533, 'sb': 0.0333}, seed-consistent False
  * sa: 0.2333 → 0.1800, win/tie/loss 9/124/17, 95% CI [-0.1200, +0.0133]
  * sb: 0.1667 → 0.2000, win/tie/loss 13/129/8, 95% CI [-0.0267, +0.0933]

### C E1-2.96M vs E1-1.60M
* **usable** pooled Δ +0.1100 (floor 0.0800) — above the floor and seed-consistent (better); seeds {'sa': 0.0733, 'sb': 0.1467}, seed-consistent True
  * sa: 0.7800 → 0.8533, win/tie/loss 21/119/10, 95% CI [+0.0000, +0.1467]
  * sb: 0.6800 → 0.8267, win/tie/loss 34/104/12, 95% CI [+0.0600, +0.2333] excludes 0
* **correct** pooled Δ +0.0200 (floor 0.0600) — inside the floor — a tie; seeds {'sa': 0.04, 'sb': 0.0}, seed-consistent False
  * sa: 0.1733 → 0.2133, win/tie/loss 20/116/14, 95% CI [-0.0333, +0.1133]
  * sb: 0.2000 → 0.2000, win/tie/loss 11/128/11, 95% CI [-0.0600, +0.0600]

### P2-1.60M vs E1-1.60M
* **usable** pooled Δ +0.0033 (floor 0.0800) — inside the floor — a tie; seeds {'sa': -0.0667, 'sb': 0.0733}, seed-consistent False
  * sa: 0.7800 → 0.7133, win/tie/loss 14/112/24, 95% CI [-0.1467, +0.0133]
  * sb: 0.6800 → 0.7533, win/tie/loss 26/109/15, 95% CI [-0.0067, +0.1533]
* **correct** pooled Δ +0.0133 (floor 0.0600) — inside the floor — a tie; seeds {'sa': 0.06, 'sb': -0.0333}, seed-consistent False
  * sa: 0.1733 → 0.2333, win/tie/loss 16/127/7, 95% CI [+0.0000, +0.1267]
  * sb: 0.2000 → 0.1667, win/tie/loss 6/133/11, 95% CI [-0.0867, +0.0200]

## Objective × scale interaction

`(P2_2.96 − P2_1.60) − (E1_2.96 − E1_1.60)`. Four two-seed cells, so
this compounds four single draws — the direction agreement across seeds
carries more weight than the point estimate, and a nonzero value is not
by itself evidence of interaction.

| metric | better | P2 scale Δ | E1 scale Δ | interaction | per seed | consistent | claimable |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| usable_rollout_rate | higher | +0.0267 | +0.1100 | **-0.0833** | {'sa': -0.0133, 'sb': -0.1533} | True | **yes** |
| correct_overall | higher | -0.0100 | +0.0200 | **-0.0300** | {'sa': -0.0933, 'sb': 0.0333} | False | no |
| correct_given_usable | higher | -0.0235 | -0.0073 | **-0.0162** | {'sa': -0.1215, 'sb': 0.0891} | False | no |
| natural_termination_rate | higher | +0.0133 | +0.1000 | **-0.0867** | {'sa': -0.0067, 'sb': -0.1667} | True | no |
| context_limit_rate | lower | -0.0133 | -0.1000 | **+0.0867** | {'sa': 0.0067, 'sb': 0.1667} | True | no |
| severe_repetition_rate | lower | -0.0100 | -0.0933 | **+0.0833** | {'sa': -0.0, 'sb': 0.1667} | False | no |
| empty_output_rate | lower | -0.0166 | -0.0566 | **+0.0400** | {'sa': -0.0333, 'sb': 0.1133} | False | no |
| answer_parse_failure_rate_numeric | lower | -0.0666 | -0.1266 | **+0.0600** | {'sa': 0.0799, 'sb': 0.0401} | True | no |

