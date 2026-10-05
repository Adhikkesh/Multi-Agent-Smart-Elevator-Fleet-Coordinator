# LiftZero — closed-loop evaluation

Seeds: `val` (n=50); generated 2026-10-05.

Real Mesa simulator, safety invariants checked every tick. Mean over paired seeds with 95 % bootstrap CIs. `disagree %` = learned awards that differ from the shadow teacher's choice (only for runs with the shadow teacher on).

## up_peak

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| liftzero_bc | 29.7 [25.1, 35.1] | 76.9 [68.9, 86.0] | 98 | 16.2 | 667 | 1154 | 1.90 | 0 | 5.3 |
| full | 29.2 [24.3, 35.0] | 75.4 [66.5, 85.3] | 98 | 16.3 | 663 | 1162 | 1.83 | 0 | — |
| liftzero_bc_cnp | 37.7 [32.3, 43.1] | 86.4 [77.3, 96.1] | 110 | 25.2 | 641 | 906 | 2.12 | 0 | 5.7 |
| cnp_astar | 40.9 [35.6, 46.8] | 93.0 [84.4, 102.7] | 118 | 28.3 | 636 | 891 | 2.20 | 0 | — |

## down_peak

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| liftzero_bc | 23.9 [23.0, 24.7] | 69.3 [66.6, 72.0] | 118 | 8.3 | 668 | 1453 | 4.75 | 0 | 10.0 |
| full | 24.0 [23.2, 24.7] | 70.1 [67.4, 73.0] | 133 | 8.4 | 677 | 1452 | 4.14 | 0 | — |
| liftzero_bc_cnp | 22.2 [21.6, 22.8] | 63.5 [61.4, 65.8] | 109 | 6.6 | 677 | 1418 | 2.90 | 0 | 10.4 |
| cnp_astar | 22.0 [21.5, 22.5] | 63.9 [62.1, 65.7] | 107 | 6.5 | 676 | 1407 | 2.78 | 0 | — |

## two_way

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| liftzero_bc | 15.3 [14.7, 15.8] | 47.3 [45.2, 49.6] | 93 | 2.7 | 626 | 1364 | 3.55 | 0 | 9.0 |
| full | 14.4 [13.8, 14.9] | 44.1 [41.8, 46.4] | 93 | 2.3 | 595 | 1339 | 2.56 | 0 | — |
| liftzero_bc_cnp | 14.7 [14.1, 15.2] | 45.3 [43.0, 47.5] | 89 | 2.7 | 610 | 1320 | 2.53 | 0 | 9.3 |
| cnp_astar | 14.5 [14.0, 15.2] | 45.4 [42.8, 48.1] | 93 | 2.8 | 611 | 1315 | 2.31 | 0 | — |

## interfloor

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| liftzero_bc | 11.6 [11.1, 12.0] | 33.3 [31.2, 35.3] | 61 | 1.0 | 356 | 1249 | 1.92 | 0 | 8.4 |
| full | 11.3 [10.9, 11.7] | 32.0 [29.9, 33.9] | 61 | 0.8 | 357 | 1255 | 1.40 | 0 | — |
| liftzero_bc_cnp | 11.6 [11.2, 11.9] | 33.2 [31.4, 35.2] | 59 | 0.8 | 343 | 992 | 1.72 | 0 | 7.6 |
| cnp_astar | 11.8 [11.4, 12.3] | 34.0 [32.1, 35.9] | 66 | 1.1 | 342 | 987 | 1.37 | 0 | — |

## Learned vs teacher (paired)

Target: average wait within ±5 % of the teacher, p95 within ±10 %.

| regime | learned | teacher | metric | learned | teacher | Δ % [95 % CI] | n | target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| up_peak | liftzero_bc | full | avg_wait | 29.7 | 29.2 | +1.5 [-9.1, +12.9] | 50 | met |
| up_peak | liftzero_bc | full | p95_wait | 76.9 | 75.4 | +2.1 [-6.3, +11.4] | 50 | met |
| down_peak | liftzero_bc | full | avg_wait | 23.9 | 24.0 | -0.2 [-5.0, +4.7] | 50 | met |
| down_peak | liftzero_bc | full | p95_wait | 69.3 | 70.1 | -1.2 [-6.8, +4.5] | 50 | met |
| two_way | liftzero_bc | full | avg_wait | 15.3 | 14.4 | +6.2 [+0.9, +12.0] | 50 | missed |
| two_way | liftzero_bc | full | p95_wait | 47.3 | 44.1 | +7.2 [+0.8, +14.7] | 50 | met |
| interfloor | liftzero_bc | full | avg_wait | 11.6 | 11.3 | +1.8 [-2.8, +6.7] | 50 | met |
| interfloor | liftzero_bc | full | p95_wait | 33.3 | 32.0 | +4.2 [-3.4, +12.5] | 50 | met |
| up_peak | liftzero_bc_cnp | cnp_astar | avg_wait | 37.7 | 40.9 | -7.8 [-14.8, -1.5] | 50 | met |
| up_peak | liftzero_bc_cnp | cnp_astar | p95_wait | 86.4 | 93.0 | -7.2 [-12.8, -2.2] | 50 | met |
| down_peak | liftzero_bc_cnp | cnp_astar | avg_wait | 22.2 | 22.0 | +0.7 [-2.1, +3.6] | 50 | met |
| down_peak | liftzero_bc_cnp | cnp_astar | p95_wait | 63.5 | 63.9 | -0.6 [-4.3, +3.5] | 50 | met |
| two_way | liftzero_bc_cnp | cnp_astar | avg_wait | 14.7 | 14.5 | +1.0 [-2.3, +4.3] | 50 | met |
| two_way | liftzero_bc_cnp | cnp_astar | p95_wait | 45.3 | 45.4 | -0.4 [-5.5, +4.6] | 50 | met |
| interfloor | liftzero_bc_cnp | cnp_astar | avg_wait | 11.6 | 11.8 | -2.2 [-5.2, +0.9] | 50 | met |
| interfloor | liftzero_bc_cnp | cnp_astar | p95_wait | 33.2 | 34.0 | -2.3 [-7.1, +2.8] | 50 | met |
