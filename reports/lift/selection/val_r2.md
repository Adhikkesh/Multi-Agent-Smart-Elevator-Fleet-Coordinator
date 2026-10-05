# LiftZero — closed-loop evaluation

Seeds: `val` (n=50); generated 2026-10-05.

Real Mesa simulator, safety invariants checked every tick. Mean over paired seeds with 95 % bootstrap CIs. `disagree %` = learned awards that differ from the shadow teacher's choice (only for runs with the shadow teacher on).

## up_peak

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| liftzero_bc | 28.8 [24.3, 33.8] | 74.9 [67.5, 83.2] | 95 | 15.3 | 663 | 1154 | 0.75 | 0 | 5.1 |
| full | 29.2 [24.3, 35.0] | 75.4 [66.5, 85.3] | 98 | 16.3 | 663 | 1162 | 0.80 | 0 | — |
| liftzero_bc_cnp | 38.6 [33.3, 44.2] | 90.3 [80.7, 100.9] | 119 | 25.6 | 630 | 894 | 0.94 | 0 | 4.7 |
| cnp_astar | 40.9 [35.6, 46.8] | 93.0 [84.4, 102.7] | 118 | 28.3 | 636 | 891 | 0.86 | 0 | — |

## down_peak

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| liftzero_bc | 23.6 [22.8, 24.4] | 67.0 [64.6, 69.6] | 115 | 8.1 | 664 | 1444 | 1.89 | 0 | 10.2 |
| full | 23.9 [23.2, 24.7] | 70.2 [67.4, 73.1] | 134 | 8.4 | 678 | 1452 | 1.89 | 0 | — |
| liftzero_bc_cnp | 22.0 [21.3, 22.6] | 63.1 [60.9, 65.1] | 103 | 6.3 | 676 | 1417 | 1.24 | 0 | 9.6 |
| cnp_astar | 22.0 [21.5, 22.5] | 63.8 [62.1, 65.6] | 108 | 6.5 | 676 | 1407 | 0.88 | 0 | — |

## two_way

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| liftzero_bc | 14.5 [14.0, 15.1] | 44.8 [42.6, 47.0] | 85 | 2.4 | 614 | 1354 | 1.41 | 0 | 7.7 |
| full | 14.4 [13.8, 14.9] | 44.1 [41.8, 46.4] | 93 | 2.4 | 596 | 1340 | 1.17 | 0 | — |
| liftzero_bc_cnp | 14.4 [14.0, 14.9] | 45.0 [42.7, 47.3] | 89 | 2.4 | 611 | 1320 | 1.31 | 0 | 8.3 |
| cnp_astar | 14.6 [14.0, 15.2] | 45.6 [42.9, 48.4] | 93 | 2.8 | 611 | 1315 | 0.80 | 0 | — |

## interfloor

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| liftzero_bc | 11.2 [10.8, 11.5] | 31.2 [29.5, 32.9] | 62 | 0.6 | 352 | 1250 | 0.82 | 0 | 7.0 |
| full | 11.3 [10.9, 11.7] | 32.0 [29.9, 33.9] | 61 | 0.8 | 357 | 1255 | 0.82 | 0 | — |
| liftzero_bc_cnp | 11.6 [11.2, 12.0] | 33.3 [31.3, 35.4] | 59 | 0.7 | 343 | 993 | 0.70 | 0 | 6.5 |
| cnp_astar | 11.8 [11.4, 12.3] | 34.0 [32.1, 35.9] | 66 | 1.1 | 342 | 987 | 0.50 | 0 | — |

## Learned vs teacher (paired)

Target: average wait within ±5 % of the teacher, p95 within ±10 %.

| regime | learned | teacher | metric | learned | teacher | Δ % [95 % CI] | n | target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| up_peak | liftzero_bc | full | avg_wait | 28.8 | 29.2 | -1.7 [-12.0, +10.3] | 50 | met |
| up_peak | liftzero_bc | full | p95_wait | 74.9 | 75.4 | -0.5 [-8.5, +8.4] | 50 | met |
| down_peak | liftzero_bc | full | avg_wait | 23.6 | 23.9 | -1.2 [-6.0, +3.2] | 50 | met |
| down_peak | liftzero_bc | full | p95_wait | 67.0 | 70.2 | -4.6 [-10.1, +1.0] | 50 | met |
| two_way | liftzero_bc | full | avg_wait | 14.5 | 14.4 | +1.1 [-3.8, +6.1] | 50 | met |
| two_way | liftzero_bc | full | p95_wait | 44.8 | 44.1 | +1.6 [-4.9, +8.8] | 50 | met |
| interfloor | liftzero_bc | full | avg_wait | 11.2 | 11.3 | -1.6 [-5.5, +2.2] | 50 | met |
| interfloor | liftzero_bc | full | p95_wait | 31.2 | 32.0 | -2.3 [-8.5, +4.7] | 50 | met |
| up_peak | liftzero_bc_cnp | cnp_astar | avg_wait | 38.6 | 40.9 | -5.7 [-12.8, +1.3] | 50 | met |
| up_peak | liftzero_bc_cnp | cnp_astar | p95_wait | 90.3 | 93.0 | -2.9 [-8.6, +3.7] | 50 | met |
| down_peak | liftzero_bc_cnp | cnp_astar | avg_wait | 22.0 | 22.0 | -0.3 [-2.9, +2.3] | 50 | met |
| down_peak | liftzero_bc_cnp | cnp_astar | p95_wait | 63.1 | 63.8 | -1.2 [-5.0, +2.7] | 50 | met |
| two_way | liftzero_bc_cnp | cnp_astar | avg_wait | 14.4 | 14.6 | -0.7 [-3.3, +1.9] | 50 | met |
| two_way | liftzero_bc_cnp | cnp_astar | p95_wait | 45.0 | 45.6 | -1.3 [-5.9, +3.8] | 50 | met |
| interfloor | liftzero_bc_cnp | cnp_astar | avg_wait | 11.6 | 11.8 | -1.8 [-5.2, +1.7] | 50 | met |
| interfloor | liftzero_bc_cnp | cnp_astar | p95_wait | 33.3 | 34.0 | -2.0 [-6.8, +3.0] | 50 | met |
