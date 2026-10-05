# LiftZero — closed-loop evaluation

Seeds: `test` (n=100); generated 2026-10-05.

Real Mesa simulator, safety invariants checked every tick. Mean over paired seeds with 95 % bootstrap CIs. `disagree %` = learned awards that differ from the shadow teacher's choice (only for runs with the shadow teacher on).

## up_peak

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| liftzero_bc | 31.6 [27.8, 35.8] | 79.7 [72.6, 87.4] | 102 | 19.0 | 656 | 1139 | 1.41 | 0 | 5.1 |
| liftzero_bc_cnp | 43.4 [39.0, 48.1] | 101.8 [94.2, 109.8] | 125 | 29.0 | 637 | 882 | 0.88 | 0 | 5.9 |

## down_peak

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| liftzero_bc | 24.0 [23.4, 24.7] | 69.1 [66.9, 71.2] | 122 | 8.5 | 671 | 1454 | 2.31 | 0 | 9.9 |
| liftzero_bc_cnp | 21.9 [21.4, 22.3] | 62.8 [61.4, 64.2] | 102 | 6.4 | 688 | 1414 | 1.14 | 0 | 9.5 |

## two_way

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| liftzero_bc | 14.9 [14.5, 15.3] | 45.8 [44.2, 47.5] | 88 | 2.5 | 618 | 1353 | 2.11 | 0 | 8.4 |
| liftzero_bc_cnp | 14.7 [14.3, 15.1] | 45.7 [44.1, 47.3] | 89 | 2.7 | 622 | 1324 | 1.02 | 0 | 7.8 |

## interfloor

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| liftzero_bc | 11.1 [10.8, 11.4] | 31.0 [29.6, 32.5] | 61 | 0.9 | 344 | 1236 | 1.30 | 0 | 7.6 |
| liftzero_bc_cnp | 11.6 [11.3, 12.0] | 34.0 [32.5, 35.5] | 65 | 1.0 | 348 | 993 | 0.67 | 0 | 6.8 |

## car_breakdown

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| liftzero_bc | 14.6 [13.4, 16.0] | 48.2 [45.2, 51.4] | 70 | 3.0 | 548 | 1181 | 0.97 | 0 | 4.0 |
| liftzero_bc_cnp | 24.2 [22.4, 26.1] | 65.3 [61.6, 69.2] | 90 | 10.5 | 538 | 852 | 0.77 | 0 | 5.0 |

## demo_story

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| liftzero_bc | 37.8 [35.4, 40.3] | 97.9 [93.1, 103.0] | 123 | 24.6 | 603 | 388 | 1.20 | 0 | 20.5 |
| liftzero_bc_cnp | 40.3 [38.1, 42.5] | 100.2 [95.7, 105.3] | 127 | 29.5 | 607 | 356 | 1.08 | 3 | 21.4 |

## evening_down_peak

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| liftzero_bc | 21.2 [20.6, 21.8] | 64.2 [62.2, 66.1] | 109 | 6.6 | 537 | 1315 | 1.71 | 0 | 8.1 |
| liftzero_bc_cnp | 19.8 [19.4, 20.2] | 58.4 [57.0, 59.8] | 93 | 4.9 | 539 | 1236 | 0.93 | 0 | 8.2 |

## fire_emergency

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| liftzero_bc | 67.3 [65.4, 69.2] | 258.8 [254.8, 262.9] | 345 | 32.8 | 481 | 918 | 1.43 | 6 | 19.9 |
| liftzero_bc_cnp | 68.0 [66.1, 69.9] | 261.6 [257.4, 265.6] | 349 | 32.5 | 492 | 894 | 1.05 | 0 | 19.8 |

## interfloor_light

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| liftzero_bc | 8.7 [8.5, 8.9] | 22.3 [21.2, 23.6] | 41 | 0.2 | 211 | 943 | 0.57 | 0 | 6.1 |
| liftzero_bc_cnp | 9.7 [9.4, 10.0] | 25.5 [24.3, 26.7] | 44 | 0.3 | 207 | 646 | 0.47 | 0 | 4.5 |

## lunch_two_way

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| liftzero_bc | 14.1 [13.7, 14.4] | 42.9 [41.3, 44.5] | 79 | 2.0 | 521 | 1299 | 1.25 | 0 | 8.3 |
| liftzero_bc_cnp | 14.0 [13.6, 14.4] | 44.0 [42.1, 45.7] | 81 | 2.4 | 523 | 1202 | 0.88 | 0 | 7.6 |

## morning_up_peak

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| liftzero_bc | 25.2 [21.9, 28.7] | 68.3 [62.0, 75.2] | 90 | 13.0 | 538 | 1126 | 0.81 | 0 | 5.1 |
| liftzero_bc_cnp | 39.1 [35.0, 43.4] | 93.2 [85.2, 101.5] | 119 | 26.2 | 533 | 830 | 0.75 | 0 | 5.5 |

## priority_passenger

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| liftzero_bc | 12.9 [12.5, 13.2] | 39.5 [37.9, 41.1] | 77 | 1.6 | 511 | 1269 | 1.24 | 0 | 8.0 |
| liftzero_bc_cnp | 13.0 [12.7, 13.2] | 39.5 [38.4, 40.8] | 77 | 1.8 | 524 | 1227 | 0.86 | 0 | 8.1 |

## stress_scale

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| liftzero_bc | 17.6 [17.2, 18.0] | 68.4 [66.9, 69.9] | 180 | 7.1 | 368 | 10189 | 2.10 | 0 | 8.0 |
| liftzero_bc_cnp | 22.9 [22.4, 23.5] | 74.5 [72.4, 76.6] | 144 | 9.6 | 365 | 6667 | 1.57 | 0 | 7.3 |
