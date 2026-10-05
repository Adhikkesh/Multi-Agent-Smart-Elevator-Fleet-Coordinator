# LiftZero — closed-loop evaluation

Seeds: `test` (n=100); generated 2026-10-05.

Real Mesa simulator, safety invariants checked every tick. Mean over paired seeds with 95 % bootstrap CIs. `disagree %` = learned awards that differ from the shadow teacher's choice (only for runs with the shadow teacher on).

## fire_emergency

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nearest_car | 70.6 [68.6, 72.7] | 262.0 [258.1, 266.0] | 345 | 36.3 | 481 | 988 | 0.42 | 0 | — |
| collective | 69.7 [67.4, 72.2] | 259.0 [254.9, 263.1] | 334 | 34.5 | 483 | 985 | 0.44 | 0 | — |
| cnp_astar | 71.8 [69.1, 74.8] | 269.2 [263.9, 274.7] | 372 | 34.3 | 486 | 896 | 0.74 | 0 | — |
| full | 69.4 [67.3, 71.7] | 265.2 [261.4, 269.2] | 363 | 33.4 | 477 | 921 | 1.02 | 0 | — |
| liftzero_bc | 67.1 [65.2, 69.0] | 258.7 [254.6, 262.8] | 346 | 32.7 | 483 | 921 | 1.19 | 0 | 20.0 |
| liftzero_bc_cnp | 67.9 [66.1, 69.8] | 261.7 [257.4, 265.7] | 348 | 32.6 | 492 | 894 | 0.94 | 0 | 19.8 |

## demo_story

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nearest_car | 46.6 [43.7, 49.4] | 122.1 [114.2, 130.0] | 153 | 33.3 | 570 | 387 | 0.35 | 0 | — |
| collective | 40.2 [37.8, 42.8] | 116.1 [108.6, 124.4] | 145 | 26.9 | 580 | 400 | 0.36 | 0 | — |
| cnp_astar | 38.8 [36.7, 40.9] | 99.5 [94.7, 104.4] | 129 | 27.2 | 605 | 355 | 0.69 | 0 | — |
| full | 35.3 [33.2, 37.7] | 91.8 [87.5, 96.9] | 119 | 21.8 | 604 | 385 | 0.79 | 0 | — |
| liftzero_bc | 37.8 [35.4, 40.3] | 97.9 [93.1, 103.0] | 123 | 24.6 | 603 | 388 | 1.04 | 0 | 20.9 |
| liftzero_bc_cnp | 40.3 [38.1, 42.5] | 100.2 [95.7, 105.3] | 127 | 29.5 | 607 | 356 | 1.00 | 0 | 21.5 |

## Learned vs teacher (paired)

Target: average wait within ±5 % of the teacher, p95 within ±10 %.

| regime | learned | teacher | metric | learned | teacher | Δ % [95 % CI] | n | target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fire_emergency | liftzero_bc | full | avg_wait | 67.1 | 69.4 | -3.3 [-6.8, +0.2] | 100 | met |
| fire_emergency | liftzero_bc | full | p95_wait | 258.7 | 265.2 | -2.5 [-4.1, -0.9] | 100 | met |
| demo_story | liftzero_bc | full | avg_wait | 37.8 | 35.3 | +7.0 [+0.1, +14.1] | 100 | missed |
| demo_story | liftzero_bc | full | p95_wait | 97.9 | 91.8 | +6.6 [+0.6, +13.1] | 100 | met |
| fire_emergency | liftzero_bc_cnp | cnp_astar | avg_wait | 67.9 | 71.8 | -5.4 [-8.4, -3.0] | 100 | met |
| fire_emergency | liftzero_bc_cnp | cnp_astar | p95_wait | 261.7 | 269.2 | -2.8 [-4.5, -1.4] | 100 | met |
| demo_story | liftzero_bc_cnp | cnp_astar | avg_wait | 40.3 | 38.8 | +3.9 [-0.6, +8.4] | 100 | met |
| demo_story | liftzero_bc_cnp | cnp_astar | p95_wait | 100.2 | 99.5 | +0.7 [-3.8, +6.0] | 100 | met |
