# LiftZero — closed-loop evaluation

Seeds: `test` (n=20); generated 2026-10-05.

Real Mesa simulator, safety invariants checked every tick. Mean over paired seeds with 95 % bootstrap CIs. `disagree %` = learned awards that differ from the shadow teacher's choice (only for runs with the shadow teacher on).

## up_peak

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| liftzero_bc_mcts | 31.9 [21.9, 44.8] | 79.0 [59.8, 103.5] | 102 | 17.3 | 678 | 1179 | 4.56 | 0 | — |
| liftzero_bc | 36.3 [26.3, 46.8] | 88.4 [71.0, 107.3] | 109 | 22.7 | 661 | 1101 | 0.73 | 0 | — |
| full | 32.3 [21.5, 43.8] | 79.4 [60.7, 98.3] | 101 | 19.7 | 660 | 1140 | 0.77 | 0 | — |

## down_peak

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| liftzero_bc_mcts | 23.5 [22.5, 24.4] | 66.4 [63.1, 70.1] | 124 | 7.8 | 690 | 1465 | 16.07 | 0 | — |
| liftzero_bc | 24.3 [22.7, 26.1] | 71.1 [65.0, 77.8] | 125 | 8.6 | 663 | 1460 | 2.02 | 0 | — |
| full | 23.4 [22.3, 24.5] | 68.0 [64.5, 71.9] | 129 | 8.1 | 679 | 1455 | 2.33 | 0 | — |

## two_way

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| liftzero_bc_mcts | 13.9 [13.4, 14.5] | 43.7 [40.7, 46.6] | 84 | 1.8 | 613 | 1351 | 11.56 | 0 | — |
| liftzero_bc | 14.2 [13.5, 14.9] | 43.9 [40.7, 46.9] | 81 | 2.3 | 606 | 1349 | 1.39 | 0 | — |
| full | 14.4 [13.7, 15.1] | 43.3 [40.4, 46.5] | 86 | 2.2 | 610 | 1362 | 1.08 | 0 | — |

## interfloor

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| liftzero_bc_mcts | 10.7 [10.3, 11.1] | 31.0 [27.9, 34.3] | 58 | 0.7 | 326 | 1224 | 6.19 | 0 | — |
| liftzero_bc | 10.8 [10.1, 11.4] | 28.9 [26.1, 32.0] | 57 | 0.7 | 332 | 1238 | 0.73 | 0 | — |
| full | 11.0 [10.3, 11.8] | 31.3 [28.2, 35.0] | 66 | 1.1 | 331 | 1222 | 0.49 | 0 | — |

## Learned vs teacher (paired)

Target: average wait within ±5 % of the teacher, p95 within ±10 %.

| regime | learned | teacher | metric | learned | teacher | Δ % [95 % CI] | n | target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| up_peak | liftzero_bc | full | avg_wait | 36.3 | 32.3 | +12.3 [-10.7, +45.2] | 20 | missed |
| up_peak | liftzero_bc | full | p95_wait | 88.4 | 79.4 | +11.4 [-7.2, +35.5] | 20 | missed |
| down_peak | liftzero_bc | full | avg_wait | 24.3 | 23.4 | +3.9 [-5.0, +13.2] | 20 | met |
| down_peak | liftzero_bc | full | p95_wait | 71.1 | 68.0 | +4.5 [-7.8, +17.2] | 20 | met |
| two_way | liftzero_bc | full | avg_wait | 14.2 | 14.4 | -1.1 [-7.3, +5.0] | 20 | met |
| two_way | liftzero_bc | full | p95_wait | 43.9 | 43.3 | +1.3 [-7.4, +11.3] | 20 | met |
| interfloor | liftzero_bc | full | avg_wait | 10.8 | 11.0 | -2.6 [-10.1, +5.4] | 20 | met |
| interfloor | liftzero_bc | full | p95_wait | 28.9 | 31.3 | -7.5 [-17.3, +2.4] | 20 | met |
