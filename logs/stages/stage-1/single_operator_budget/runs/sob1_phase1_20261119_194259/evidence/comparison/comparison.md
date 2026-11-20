# SOB1 phase 1: teacher vs each untrained arm (8,192 tokens, eval-set-1)

_descriptive only: the teacher minus each untrained arm, item by item, under one protocol; no arm is ranked, eliminated or kept by it -- all three go to recovery and are compared after it_

## 思考模式 (H-T)

| arm | items (groups) | teacher % | arm % | teacher − arm, pp | 95% interval, pp |
| --- | ---: | ---: | ---: | ---: | --- |
| 仅残差宽度 (width 1280) | 570 (558) | 85.79 | 0.0 | 85.79 | 82.87 to 88.58 |
| 仅 FFN + 注意力 (FFN 3840, 16Q/8KV) | 570 (558) | 85.79 | 0.0 | 85.79 | 82.87 to 88.58 |
| 仅深度 (16 layers) | 570 (558) | 85.79 | 0.0 | 85.79 | 82.87 to 88.58 |

Static fluid formal core (primary items). Interactive episodes and supporting units, each apart:

| arm | unit | items | teacher % | arm % | teacher − arm, pp | 95% interval, pp |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 仅残差宽度 (width 1280) | interactive episodes | 144 | 25.0 | 0.0 | 25.0 | 18.06 to 32.64 |
| 仅残差宽度 (width 1280) | stage3_capability_846__answerability_paired ¹ | 120 | 73.33 | 0.0 | 73.33 | 65.0 to 80.83 |
| 仅残差宽度 (width 1280) | stage3_capability_846__gsm8k | 100 | 89.0 | 0.0 | 89.0 | 83.0 to 95.0 |
| 仅残差宽度 (width 1280) | stage3_capability_846__knowledge | 150 | 55.33 | 0.0 | 55.33 | 47.33 to 63.33 |
| 仅残差宽度 (width 1280) | stage3_capability_846__math_verified | 100 | 76.0 | 0.0 | 76.0 | 67.0 to 84.0 |
| 仅残差宽度 (width 1280) | stage3_capability_846__multihop | 100 | 80.0 | 0.0 | 80.0 | 72.0 to 87.0 |
| 仅残差宽度 (width 1280) | stage3_capability_846__rag | 100 | 70.0 | 0.0 | 70.0 | 61.0 to 79.0 |
| 仅残差宽度 (width 1280) | stage3_capability_846__safety_paired ¹ | 100 | 71.0 | 0.0 | 71.0 | 62.0 to 80.0 |
| 仅残差宽度 (width 1280) | stage3_legacy_150__items | 150 | 94.67 | 0.0 | 94.67 | 90.67 to 98.0 |
| 仅 FFN + 注意力 (FFN 3840, 16Q/8KV) | interactive episodes | 144 | 25.0 | 0.0 | 25.0 | 18.06 to 32.64 |
| 仅 FFN + 注意力 (FFN 3840, 16Q/8KV) | stage3_capability_846__answerability_paired ¹ | 120 | 73.33 | 0.0 | 73.33 | 65.0 to 80.83 |
| 仅 FFN + 注意力 (FFN 3840, 16Q/8KV) | stage3_capability_846__gsm8k | 100 | 89.0 | 0.0 | 89.0 | 83.0 to 95.0 |
| 仅 FFN + 注意力 (FFN 3840, 16Q/8KV) | stage3_capability_846__knowledge | 150 | 55.33 | 0.0 | 55.33 | 47.33 to 63.33 |
| 仅 FFN + 注意力 (FFN 3840, 16Q/8KV) | stage3_capability_846__math_verified | 100 | 76.0 | 0.0 | 76.0 | 67.0 to 84.0 |
| 仅 FFN + 注意力 (FFN 3840, 16Q/8KV) | stage3_capability_846__multihop | 100 | 80.0 | 0.0 | 80.0 | 72.0 to 87.0 |
| 仅 FFN + 注意力 (FFN 3840, 16Q/8KV) | stage3_capability_846__rag | 100 | 70.0 | 0.0 | 70.0 | 61.0 to 79.0 |
| 仅 FFN + 注意力 (FFN 3840, 16Q/8KV) | stage3_capability_846__safety_paired ¹ | 100 | 71.0 | 0.0 | 71.0 | 62.0 to 80.0 |
| 仅 FFN + 注意力 (FFN 3840, 16Q/8KV) | stage3_legacy_150__items | 150 | 94.67 | 0.0 | 94.67 | 90.67 to 98.0 |
| 仅深度 (16 layers) | interactive episodes | 144 | 25.0 | 0.0 | 25.0 | 18.06 to 32.64 |
| 仅深度 (16 layers) | stage3_capability_846__answerability_paired ¹ | 120 | 73.33 | 0.0 | 73.33 | 65.0 to 80.83 |
| 仅深度 (16 layers) | stage3_capability_846__gsm8k | 100 | 89.0 | 0.0 | 89.0 | 83.0 to 95.0 |
| 仅深度 (16 layers) | stage3_capability_846__knowledge | 150 | 55.33 | 0.0 | 55.33 | 47.33 to 63.33 |
| 仅深度 (16 layers) | stage3_capability_846__math_verified | 100 | 76.0 | 0.0 | 76.0 | 67.0 to 84.0 |
| 仅深度 (16 layers) | stage3_capability_846__multihop | 100 | 80.0 | 0.0 | 80.0 | 72.0 to 87.0 |
| 仅深度 (16 layers) | stage3_capability_846__rag | 100 | 70.0 | 0.0 | 70.0 | 61.0 to 79.0 |
| 仅深度 (16 layers) | stage3_capability_846__safety_paired ¹ | 100 | 71.0 | 0.0 | 71.0 | 62.0 to 80.0 |
| 仅深度 (16 layers) | stage3_legacy_150__items | 150 | 94.67 | 0.0 | 94.67 | 90.67 to 98.0 |

¹ per-item correctness; the unit's registered headline is its pair accuracy, reported in each set's formal report and in comparison.json :: headlines. The behaviour set has no correctness headline and is not in this table. Intervals: 95% percentile bootstrap over the registered units of independence (the primary core) or over items, unadjusted for multiplicity.

How the turns ended (counts; a degeneration stop is the engine's repetition check, a context-limit hit is right-censored; reported apart, never a score):

| model | core turns | natural | degeneration stop | context limit | set turns | natural | degeneration stop | context limit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 教师 (teacher) | 570 | 510 | 6 | 54 | 1952 | 1803 | 15 | 134 |
| 仅残差宽度 (width 1280) | 570 | 0 | 570 | 0 | 1716 | 0 | 1716 | 0 |
| 仅 FFN + 注意力 (FFN 3840, 16Q/8KV) | 570 | 0 | 570 | 0 | 1716 | 9 | 1707 | 0 |
| 仅深度 (16 layers) | 570 | 19 | 551 | 0 | 1719 | 54 | 1665 | 0 |

## 非思考模式 (H-N)

| arm | items (groups) | teacher % | arm % | teacher − arm, pp | 95% interval, pp |
| --- | ---: | ---: | ---: | ---: | --- |
| 仅残差宽度 (width 1280) | 570 (558) | 70.53 | 0.0 | 70.53 | 66.73 to 74.3 |
| 仅 FFN + 注意力 (FFN 3840, 16Q/8KV) | 570 (558) | 70.53 | 0.0 | 70.53 | 66.73 to 74.3 |
| 仅深度 (16 layers) | 570 (558) | 70.53 | 0.0 | 70.53 | 66.73 to 74.3 |

Static fluid formal core (primary items). Interactive episodes and supporting units, each apart:

| arm | unit | items | teacher % | arm % | teacher − arm, pp | 95% interval, pp |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 仅残差宽度 (width 1280) | interactive episodes | 144 | 29.17 | 0.0 | 29.17 | 21.53 to 36.81 |
| 仅残差宽度 (width 1280) | stage3_capability_846__answerability_paired ¹ | 120 | 75.0 | 0.0 | 75.0 | 66.67 to 82.5 |
| 仅残差宽度 (width 1280) | stage3_capability_846__gsm8k | 100 | 40.0 | 0.0 | 40.0 | 31.0 to 50.0 |
| 仅残差宽度 (width 1280) | stage3_capability_846__knowledge | 150 | 53.33 | 0.0 | 53.33 | 45.33 to 61.33 |
| 仅残差宽度 (width 1280) | stage3_capability_846__math_verified | 100 | 77.0 | 0.0 | 77.0 | 68.0 to 85.0 |
| 仅残差宽度 (width 1280) | stage3_capability_846__multihop | 100 | 75.0 | 0.0 | 75.0 | 66.0 to 83.0 |
| 仅残差宽度 (width 1280) | stage3_capability_846__rag | 100 | 81.0 | 0.0 | 81.0 | 73.0 to 88.0 |
| 仅残差宽度 (width 1280) | stage3_capability_846__safety_paired ¹ | 100 | 81.0 | 0.0 | 81.0 | 73.0 to 88.0 |
| 仅残差宽度 (width 1280) | stage3_legacy_150__items | 150 | 73.33 | 0.0 | 73.33 | 66.0 to 80.0 |
| 仅 FFN + 注意力 (FFN 3840, 16Q/8KV) | interactive episodes | 144 | 29.17 | 0.0 | 29.17 | 21.53 to 36.81 |
| 仅 FFN + 注意力 (FFN 3840, 16Q/8KV) | stage3_capability_846__answerability_paired ¹ | 120 | 75.0 | 0.0 | 75.0 | 66.67 to 82.5 |
| 仅 FFN + 注意力 (FFN 3840, 16Q/8KV) | stage3_capability_846__gsm8k | 100 | 40.0 | 0.0 | 40.0 | 31.0 to 50.0 |
| 仅 FFN + 注意力 (FFN 3840, 16Q/8KV) | stage3_capability_846__knowledge | 150 | 53.33 | 0.0 | 53.33 | 45.33 to 61.33 |
| 仅 FFN + 注意力 (FFN 3840, 16Q/8KV) | stage3_capability_846__math_verified | 100 | 77.0 | 0.0 | 77.0 | 68.0 to 85.0 |
| 仅 FFN + 注意力 (FFN 3840, 16Q/8KV) | stage3_capability_846__multihop | 100 | 75.0 | 0.0 | 75.0 | 66.0 to 83.0 |
| 仅 FFN + 注意力 (FFN 3840, 16Q/8KV) | stage3_capability_846__rag | 100 | 81.0 | 0.0 | 81.0 | 73.0 to 88.0 |
| 仅 FFN + 注意力 (FFN 3840, 16Q/8KV) | stage3_capability_846__safety_paired ¹ | 100 | 81.0 | 0.0 | 81.0 | 73.0 to 88.0 |
| 仅 FFN + 注意力 (FFN 3840, 16Q/8KV) | stage3_legacy_150__items | 150 | 73.33 | 0.67 | 72.67 | 65.33 to 79.33 |
| 仅深度 (16 layers) | interactive episodes | 144 | 29.17 | 0.0 | 29.17 | 21.53 to 36.81 |
| 仅深度 (16 layers) | stage3_capability_846__answerability_paired ¹ | 120 | 75.0 | 0.0 | 75.0 | 66.67 to 82.5 |
| 仅深度 (16 layers) | stage3_capability_846__gsm8k | 100 | 40.0 | 0.0 | 40.0 | 31.0 to 50.0 |
| 仅深度 (16 layers) | stage3_capability_846__knowledge | 150 | 53.33 | 0.0 | 53.33 | 45.33 to 61.33 |
| 仅深度 (16 layers) | stage3_capability_846__math_verified | 100 | 77.0 | 0.0 | 77.0 | 68.0 to 85.0 |
| 仅深度 (16 layers) | stage3_capability_846__multihop | 100 | 75.0 | 0.0 | 75.0 | 66.0 to 83.0 |
| 仅深度 (16 layers) | stage3_capability_846__rag | 100 | 81.0 | 0.0 | 81.0 | 73.0 to 88.0 |
| 仅深度 (16 layers) | stage3_capability_846__safety_paired ¹ | 100 | 81.0 | 0.0 | 81.0 | 73.0 to 88.0 |
| 仅深度 (16 layers) | stage3_legacy_150__items | 150 | 73.33 | 1.33 | 72.0 | 64.67 to 79.33 |

¹ per-item correctness; the unit's registered headline is its pair accuracy, reported in each set's formal report and in comparison.json :: headlines. The behaviour set has no correctness headline and is not in this table. Intervals: 95% percentile bootstrap over the registered units of independence (the primary core) or over items, unadjusted for multiplicity.

How the turns ended (counts; a degeneration stop is the engine's repetition check, a context-limit hit is right-censored; reported apart, never a score):

| model | core turns | natural | degeneration stop | context limit | set turns | natural | degeneration stop | context limit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 教师 (teacher) | 570 | 566 | 4 | 0 | 2575 | 2554 | 21 | 0 |
| 仅残差宽度 (width 1280) | 570 | 0 | 570 | 0 | 1716 | 0 | 1716 | 0 |
| 仅 FFN + 注意力 (FFN 3840, 16Q/8KV) | 570 | 2 | 568 | 0 | 1716 | 7 | 1709 | 0 |
| 仅深度 (16 layers) | 570 | 21 | 549 | 0 | 1719 | 56 | 1663 | 0 |

