# LiftZero — closed-loop evaluation

Seeds: `val` (n=50); generated 2026-10-05.

Real Mesa simulator, safety invariants checked every tick. Mean over paired seeds with 95 % bootstrap CIs. `disagree %` = learned awards that differ from the shadow teacher's choice (only for runs with the shadow teacher on).

## up_peak

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| liftzero_bc | 29.1 [24.4, 34.4] | 74.8 [66.6, 83.4] | 97 | 15.5 | 663 | 1153 | 1.15 | 0 | 5.4 |
| full | 29.2 [24.3, 35.0] | 75.4 [66.5, 85.3] | 98 | 16.3 | 663 | 1162 | 1.75 | 0 | — |
| liftzero_bc_cnp | 40.3 [34.8, 45.8] | 91.3 [82.2, 100.7] | 117 | 27.8 | 632 | 879 | 0.78 | 0 | 5.3 |
| cnp_astar | 40.9 [35.6, 46.8] | 93.0 [84.4, 102.7] | 118 | 28.3 | 636 | 891 | 0.68 | 0 | — |

## down_peak

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| liftzero_bc | 24.7 [23.7, 25.6] | 72.2 [69.2, 75.4] | 127 | 9.2 | 681 | 1452 | 2.80 | 0 | 10.3 |
| full | 24.0 [23.2, 24.7] | 70.3 [67.5, 73.2] | 134 | 8.4 | 677 | 1452 | 2.00 | 0 | — |
| liftzero_bc_cnp | 21.9 [21.3, 22.4] | 62.5 [60.4, 64.3] | 104 | 6.0 | 677 | 1418 | 1.32 | 0 | 9.8 |
| cnp_astar | 22.0 [21.5, 22.5] | 63.8 [62.1, 65.6] | 108 | 6.5 | 676 | 1407 | 1.94 | 0 | — |

## two_way

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| liftzero_bc | 14.8 [14.3, 15.3] | 46.3 [43.9, 48.8] | 87 | 2.6 | 619 | 1361 | 1.74 | 0 | 9.3 |
| full | 14.4 [13.8, 15.0] | 44.3 [41.9, 46.6] | 93 | 2.4 | 596 | 1340 | 1.87 | 0 | — |
| liftzero_bc_cnp | 14.6 [14.1, 15.2] | 45.8 [43.3, 48.2] | 88 | 2.6 | 610 | 1311 | 2.22 | 0 | 8.3 |
| cnp_astar | 14.5 [14.0, 15.2] | 45.4 [42.8, 48.1] | 93 | 2.8 | 611 | 1315 | 0.77 | 0 | — |

## interfloor

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| liftzero_bc | 11.0 [10.6, 11.4] | 30.6 [29.1, 32.3] | 56 | 0.6 | 355 | 1246 | 0.98 | 0 | 7.6 |
| full | 11.3 [10.9, 11.7] | 32.0 [29.9, 33.9] | 61 | 0.8 | 357 | 1255 | 0.64 | 0 | — |
| liftzero_bc_cnp | 11.5 [11.0, 11.9] | 33.0 [30.9, 35.1] | 59 | 0.7 | 343 | 991 | 0.37 | 0 | 6.9 |
| cnp_astar | 11.8 [11.4, 12.3] | 34.0 [32.1, 35.9] | 66 | 1.1 | 342 | 987 | 0.39 | 0 | — |

## Learned vs teacher (paired)

Target: average wait within ±5 % of the teacher, p95 within ±10 %.

| regime | learned | teacher | metric | learned | teacher | Δ % [95 % CI] | n | target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| up_peak | liftzero_bc | full | avg_wait | 29.1 | 29.2 | -0.3 [-12.4, +13.4] | 50 | met |
| up_peak | liftzero_bc | full | p95_wait | 74.8 | 75.4 | -0.7 [-10.5, +10.3] | 50 | met |
| down_peak | liftzero_bc | full | avg_wait | 24.7 | 24.0 | +2.9 [-1.9, +7.7] | 50 | met |
| down_peak | liftzero_bc | full | p95_wait | 72.2 | 70.3 | +2.8 [-2.2, +8.3] | 50 | met |
| two_way | liftzero_bc | full | avg_wait | 14.8 | 14.4 | +2.8 [-1.2, +7.0] | 50 | met |
| two_way | liftzero_bc | full | p95_wait | 46.3 | 44.3 | +4.5 [-1.8, +11.5] | 50 | met |
| interfloor | liftzero_bc | full | avg_wait | 11.0 | 11.3 | -2.9 [-6.4, +0.8] | 50 | met |
| interfloor | liftzero_bc | full | p95_wait | 30.6 | 32.0 | -4.2 [-10.1, +2.0] | 50 | met |
| up_peak | liftzero_bc_cnp | cnp_astar | avg_wait | 40.3 | 40.9 | -1.6 [-8.9, +5.9] | 50 | met |
| up_peak | liftzero_bc_cnp | cnp_astar | p95_wait | 91.3 | 93.0 | -1.9 [-7.1, +3.5] | 50 | met |
| down_peak | liftzero_bc_cnp | cnp_astar | avg_wait | 21.9 | 22.0 | -0.8 [-3.2, +1.7] | 50 | met |
| down_peak | liftzero_bc_cnp | cnp_astar | p95_wait | 62.5 | 63.8 | -2.1 [-5.7, +1.4] | 50 | met |
| two_way | liftzero_bc_cnp | cnp_astar | avg_wait | 14.6 | 14.5 | +0.8 [-2.1, +3.8] | 50 | met |
| two_way | liftzero_bc_cnp | cnp_astar | p95_wait | 45.8 | 45.4 | +0.8 [-4.0, +5.2] | 50 | met |
| interfloor | liftzero_bc_cnp | cnp_astar | avg_wait | 11.5 | 11.8 | -2.8 [-5.8, +0.2] | 50 | met |
| interfloor | liftzero_bc_cnp | cnp_astar | p95_wait | 33.0 | 34.0 | -2.9 [-7.7, +1.9] | 50 | met |
