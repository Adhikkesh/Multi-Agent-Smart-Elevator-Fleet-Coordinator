# LiftZero — closed-loop TEST evaluation (shipped model)

Test seeds 8500-8599 (100 paired seeds per regime), 4 Phase 3 regimes + 9 scenario YAMLs, model models/liftzero_bc_v1.onnx (DAgger round 2). Evaluated once, after selection on validation seeds. Fire scenarios re-run after the fire-recall fix (see PHASE4_IMITATION.md §9).

Real Mesa simulator, safety invariants checked every tick. Mean over paired seeds with 95 % bootstrap CIs. `disagree %` = learned awards that differ from the shadow teacher's choice (only for runs with the shadow teacher on).

## car_breakdown

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nearest_car | 30.6 [28.1, 33.4] | 77.6 [73.8, 82.2] | 100 | 16.1 | 503 | 1117 | 0.36 | 0 | — |
| collective | 24.1 [22.4, 25.9] | 68.7 [65.3, 72.4] | 91 | 10.1 | 515 | 1222 | 0.40 | 0 | — |
| cnp_astar | 24.3 [22.3, 26.5] | 65.2 [61.0, 69.6] | 90 | 10.5 | 536 | 849 | 0.82 | 0 | — |
| full | 15.3 [14.1, 16.7] | 50.3 [47.4, 53.6] | 72 | 3.4 | 545 | 1169 | 0.92 | 0 | — |
| liftzero_bc | 14.6 [13.4, 16.0] | 48.2 [45.2, 51.4] | 70 | 3.0 | 548 | 1181 | 0.97 | 0 | 4.0 |
| liftzero_bc_cnp | 24.2 [22.4, 26.1] | 65.3 [61.6, 69.2] | 90 | 10.5 | 538 | 852 | 0.77 | 0 | 5.0 |

## demo_story

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nearest_car | 46.6 [43.7, 49.4] | 122.1 [114.2, 130.0] | 153 | 33.3 | 570 | 387 | 0.35 | 0 | — |
| collective | 40.2 [37.8, 42.8] | 116.1 [108.6, 124.4] | 145 | 26.9 | 580 | 400 | 0.36 | 0 | — |
| cnp_astar | 38.8 [36.7, 40.9] | 99.5 [94.7, 104.4] | 129 | 27.2 | 605 | 355 | 0.69 | 0 | — |
| full | 35.3 [33.2, 37.7] | 91.8 [87.5, 96.9] | 119 | 21.8 | 604 | 385 | 0.79 | 0 | — |
| liftzero_bc | 37.8 [35.4, 40.3] | 97.9 [93.1, 103.0] | 123 | 24.6 | 603 | 388 | 1.04 | 0 | 20.9 |
| liftzero_bc_cnp | 40.3 [38.1, 42.5] | 100.2 [95.7, 105.3] | 127 | 29.5 | 607 | 356 | 1.00 | 0 | 21.5 |

## down_peak

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nearest_car | 22.4 [21.9, 22.9] | 69.3 [68.1, 70.4] | 110 | 10.0 | 678 | 1455 | 1.06 | 0 | — |
| collective | 21.1 [18.6, 25.2] | 67.2 [59.0, 79.5] | 112 | 6.7 | 670 | 1430 | 0.58 | 0 | — |
| cnp_astar | 22.0 [21.5, 22.4] | 62.7 [61.3, 64.1] | 110 | 6.4 | 689 | 1420 | 0.91 | 0 | — |
| full | 23.7 [23.2, 24.2] | 69.4 [67.6, 71.2] | 129 | 8.3 | 672 | 1450 | 2.54 | 0 | — |
| liftzero_bc | 24.0 [23.4, 24.7] | 69.1 [66.9, 71.2] | 122 | 8.5 | 671 | 1454 | 2.31 | 0 | 9.9 |
| liftzero_bc_cnp | 21.9 [21.4, 22.3] | 62.8 [61.4, 64.2] | 102 | 6.4 | 688 | 1414 | 1.14 | 0 | 9.5 |

## evening_down_peak

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nearest_car | 21.0 [20.4, 21.6] | 67.3 [65.8, 68.8] | 109 | 9.2 | 539 | 1308 | 0.43 | 0 | — |
| collective | 17.3 [16.9, 17.8] | 56.5 [54.6, 58.2] | 88 | 4.8 | 540 | 1282 | 0.47 | 0 | — |
| cnp_astar | 20.1 [19.7, 20.5] | 59.8 [58.4, 61.4] | 104 | 5.5 | 540 | 1232 | 0.82 | 0 | — |
| full | 21.3 [20.7, 21.9] | 64.8 [62.7, 67.0] | 110 | 7.1 | 541 | 1313 | 1.66 | 0 | — |
| liftzero_bc | 21.2 [20.6, 21.8] | 64.2 [62.2, 66.1] | 109 | 6.6 | 537 | 1315 | 1.71 | 0 | 8.1 |
| liftzero_bc_cnp | 19.8 [19.4, 20.2] | 58.4 [57.0, 59.8] | 93 | 4.9 | 539 | 1236 | 0.93 | 0 | 8.2 |

## fire_emergency

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nearest_car | 70.6 [68.6, 72.7] | 262.0 [258.1, 266.0] | 345 | 36.3 | 481 | 988 | 0.42 | 0 | — |
| collective | 69.7 [67.4, 72.2] | 259.0 [254.9, 263.1] | 334 | 34.5 | 483 | 985 | 0.44 | 0 | — |
| cnp_astar | 71.8 [69.1, 74.8] | 269.2 [263.9, 274.7] | 372 | 34.3 | 486 | 896 | 0.74 | 0 | — |
| full | 69.4 [67.3, 71.7] | 265.2 [261.4, 269.2] | 363 | 33.4 | 477 | 921 | 1.02 | 0 | — |
| liftzero_bc | 67.1 [65.2, 69.0] | 258.7 [254.6, 262.8] | 346 | 32.7 | 483 | 921 | 1.19 | 0 | 20.0 |
| liftzero_bc_cnp | 67.9 [66.1, 69.8] | 261.7 [257.4, 265.7] | 348 | 32.6 | 492 | 894 | 0.94 | 0 | 19.8 |

## interfloor

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nearest_car | 12.5 [12.1, 12.9] | 38.8 [36.7, 41.1] | 69 | 2.1 | 345 | 1168 | 0.41 | 0 | — |
| collective | 11.6 [11.3, 11.9] | 32.1 [30.7, 33.6] | 62 | 1.2 | 346 | 1127 | 0.37 | 0 | — |
| cnp_astar | 11.8 [11.4, 12.1] | 34.9 [33.3, 36.5] | 67 | 1.3 | 348 | 982 | 0.58 | 0 | — |
| full | 11.0 [10.7, 11.2] | 30.9 [29.6, 32.4] | 61 | 0.8 | 343 | 1225 | 0.78 | 0 | — |
| liftzero_bc | 11.1 [10.8, 11.4] | 31.0 [29.6, 32.5] | 61 | 0.9 | 344 | 1236 | 1.30 | 0 | 7.6 |
| liftzero_bc_cnp | 11.6 [11.3, 12.0] | 34.0 [32.5, 35.5] | 65 | 1.0 | 348 | 993 | 0.67 | 0 | 6.8 |

## interfloor_light

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nearest_car | 8.9 [8.7, 9.2] | 21.5 [20.5, 22.7] | 40 | 0.3 | 206 | 767 | 0.26 | 0 | — |
| collective | 9.0 [8.8, 9.3] | 22.7 [21.8, 23.7] | 38 | 0.2 | 206 | 740 | 0.30 | 0 | — |
| cnp_astar | 9.8 [9.5, 10.1] | 26.8 [25.4, 28.2] | 47 | 0.5 | 207 | 642 | 0.44 | 0 | — |
| full | 8.7 [8.4, 9.0] | 22.2 [21.0, 23.6] | 42 | 0.3 | 211 | 943 | 0.47 | 0 | — |
| liftzero_bc | 8.7 [8.5, 8.9] | 22.3 [21.2, 23.6] | 41 | 0.2 | 211 | 943 | 0.57 | 0 | 6.1 |
| liftzero_bc_cnp | 9.7 [9.4, 10.0] | 25.5 [24.3, 26.7] | 44 | 0.3 | 207 | 646 | 0.47 | 0 | 4.5 |

## lunch_two_way

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nearest_car | 17.5 [17.0, 18.1] | 60.1 [58.3, 61.9] | 91 | 6.1 | 519 | 1354 | 0.41 | 0 | — |
| collective | 15.6 [15.2, 16.1] | 49.8 [47.8, 51.8] | 84 | 3.3 | 520 | 1339 | 0.48 | 0 | — |
| cnp_astar | 13.9 [13.5, 14.3] | 42.4 [40.8, 44.0] | 87 | 2.3 | 523 | 1196 | 0.78 | 0 | — |
| full | 13.5 [13.1, 13.9] | 41.3 [39.8, 42.8] | 82 | 1.9 | 521 | 1286 | 1.11 | 0 | — |
| liftzero_bc | 14.1 [13.7, 14.4] | 42.9 [41.3, 44.5] | 79 | 2.0 | 521 | 1299 | 1.25 | 0 | 8.3 |
| liftzero_bc_cnp | 14.0 [13.6, 14.4] | 44.0 [42.1, 45.7] | 81 | 2.4 | 523 | 1202 | 0.88 | 0 | 7.6 |

## morning_up_peak

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nearest_car | 39.1 [35.8, 42.7] | 97.8 [91.0, 105.1] | 121 | 25.9 | 513 | 1172 | 0.40 | 0 | — |
| collective | 35.1 [31.3, 39.0] | 90.7 [83.1, 98.5] | 113 | 21.6 | 519 | 1241 | 0.41 | 0 | — |
| cnp_astar | 38.5 [34.7, 42.7] | 94.4 [86.7, 102.6] | 120 | 25.5 | 532 | 825 | 0.80 | 0 | — |
| full | 23.1 [19.9, 26.4] | 63.4 [57.2, 70.1] | 84 | 11.2 | 539 | 1136 | 0.76 | 0 | — |
| liftzero_bc | 25.2 [21.9, 28.7] | 68.3 [62.0, 75.2] | 90 | 13.0 | 538 | 1126 | 0.81 | 0 | 5.1 |
| liftzero_bc_cnp | 39.1 [35.0, 43.4] | 93.2 [85.2, 101.5] | 119 | 26.2 | 533 | 830 | 0.75 | 0 | 5.5 |

## priority_passenger

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nearest_car | 16.1 [15.6, 16.6] | 55.8 [53.9, 57.5] | 89 | 4.5 | 512 | 1386 | 0.44 | 0 | — |
| collective | 14.8 [14.4, 15.2] | 47.2 [45.4, 49.0] | 82 | 2.9 | 513 | 1383 | 0.46 | 0 | — |
| cnp_astar | 13.3 [12.9, 13.6] | 40.5 [39.1, 42.1] | 84 | 1.8 | 523 | 1226 | 0.77 | 0 | — |
| full | 13.1 [12.8, 13.4] | 40.3 [38.8, 41.9] | 81 | 1.6 | 519 | 1279 | 1.11 | 0 | — |
| liftzero_bc | 12.9 [12.5, 13.2] | 39.5 [37.9, 41.1] | 77 | 1.6 | 511 | 1269 | 1.24 | 0 | 8.0 |
| liftzero_bc_cnp | 13.0 [12.7, 13.2] | 39.5 [38.4, 40.8] | 77 | 1.8 | 524 | 1227 | 0.86 | 0 | 8.1 |

## stress_scale

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nearest_car | 26.2 [25.7, 26.9] | 93.7 [91.6, 96.5] | 167 | 16.1 | 365 | 7568 | 0.97 | 0 | — |
| collective | 16.9 [16.5, 17.2] | 56.2 [54.9, 57.5] | 116 | 4.4 | 365 | 9075 | 1.01 | 0 | — |
| cnp_astar | 23.1 [22.5, 23.7] | 75.6 [73.3, 78.3] | 155 | 9.9 | 365 | 6661 | 1.59 | 0 | — |
| full | 17.2 [16.8, 17.5] | 67.5 [66.1, 69.0] | 178 | 6.8 | 364 | 10132 | 2.04 | 0 | — |
| liftzero_bc | 17.6 [17.2, 18.0] | 68.4 [66.9, 69.9] | 180 | 7.1 | 368 | 10189 | 2.10 | 0 | 8.0 |
| liftzero_bc_cnp | 22.9 [22.4, 23.5] | 74.5 [72.4, 76.6] | 144 | 9.6 | 365 | 6667 | 1.57 | 0 | 7.3 |

## two_way

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nearest_car | 18.7 [18.2, 19.2] | 63.0 [61.6, 64.3] | 102 | 6.9 | 606 | 1442 | 0.51 | 0 | — |
| collective | 17.4 [16.9, 17.8] | 56.5 [54.8, 58.1] | 94 | 4.7 | 609 | 1444 | 0.50 | 0 | — |
| cnp_astar | 14.5 [14.2, 14.8] | 44.9 [43.2, 46.6] | 91 | 2.4 | 622 | 1320 | 0.86 | 0 | — |
| full | 14.5 [14.2, 14.8] | 45.2 [43.8, 46.8] | 90 | 2.4 | 615 | 1348 | 1.58 | 0 | — |
| liftzero_bc | 14.9 [14.5, 15.3] | 45.8 [44.2, 47.5] | 88 | 2.5 | 618 | 1353 | 2.11 | 0 | 8.4 |
| liftzero_bc_cnp | 14.7 [14.3, 15.1] | 45.7 [44.1, 47.3] | 89 | 2.7 | 622 | 1324 | 1.02 | 0 | 7.8 |

## up_peak

| strategy | avg wait (s) | p95 wait (s) | max wait | long-wait % | throughput/h | energy | ms/tick | violations | disagree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nearest_car | 56.2 [51.3, 61.2] | 124.9 [116.1, 133.9] | 149 | 39.4 | 576 | 1223 | 1.02 | 0 | — |
| collective | 47.1 [42.3, 52.0] | 110.7 [101.5, 120.3] | 136 | 31.7 | 589 | 1294 | 0.44 | 0 | — |
| cnp_astar | 42.9 [38.6, 47.4] | 100.7 [92.9, 108.5] | 124 | 28.9 | 641 | 883 | 0.80 | 0 | — |
| full | 29.9 [26.1, 33.9] | 77.8 [70.7, 85.1] | 101 | 17.5 | 655 | 1157 | 1.04 | 0 | — |
| liftzero_bc | 31.6 [27.8, 35.8] | 79.7 [72.6, 87.4] | 102 | 19.0 | 656 | 1139 | 1.41 | 0 | 5.1 |
| liftzero_bc_cnp | 43.4 [39.0, 48.1] | 101.8 [94.2, 109.8] | 125 | 29.0 | 637 | 882 | 0.88 | 0 | 5.9 |

## Learned vs teacher (paired)

Target: average wait within ±5 % of the teacher, p95 within ±10 %.

| regime | learned | teacher | metric | learned | teacher | Δ % [95 % CI] | n | target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| car_breakdown | liftzero_bc | full | avg_wait | 14.6 | 15.3 | -4.8 [-11.2, +1.9] | 100 | met |
| car_breakdown | liftzero_bc | full | p95_wait | 48.2 | 50.3 | -4.1 [-9.6, +1.4] | 100 | met |
| demo_story | liftzero_bc | full | avg_wait | 37.8 | 35.3 | +7.0 [+0.1, +14.1] | 100 | missed |
| demo_story | liftzero_bc | full | p95_wait | 97.9 | 91.8 | +6.6 [+0.6, +13.1] | 100 | met |
| down_peak | liftzero_bc | full | avg_wait | 24.0 | 23.7 | +1.4 [-2.0, +5.0] | 100 | met |
| down_peak | liftzero_bc | full | p95_wait | 69.1 | 69.4 | -0.4 [-4.5, +4.0] | 100 | met |
| evening_down_peak | liftzero_bc | full | avg_wait | 21.2 | 21.3 | -0.1 [-3.9, +3.5] | 100 | met |
| evening_down_peak | liftzero_bc | full | p95_wait | 64.2 | 64.8 | -0.9 [-5.0, +2.9] | 100 | met |
| fire_emergency | liftzero_bc | full | avg_wait | 67.1 | 69.4 | -3.3 [-6.8, +0.2] | 100 | met |
| fire_emergency | liftzero_bc | full | p95_wait | 258.7 | 265.2 | -2.5 [-4.1, -0.9] | 100 | met |
| interfloor | liftzero_bc | full | avg_wait | 11.1 | 11.0 | +1.1 [-2.0, +4.4] | 100 | met |
| interfloor | liftzero_bc | full | p95_wait | 31.0 | 30.9 | +0.3 [-4.9, +5.5] | 100 | met |
| interfloor_light | liftzero_bc | full | avg_wait | 8.7 | 8.7 | -0.1 [-2.9, +2.5] | 100 | met |
| interfloor_light | liftzero_bc | full | p95_wait | 22.3 | 22.2 | +0.8 [-4.5, +5.7] | 100 | met |
| lunch_two_way | liftzero_bc | full | avg_wait | 14.1 | 13.5 | +4.2 [+0.8, +7.9] | 100 | met |
| lunch_two_way | liftzero_bc | full | p95_wait | 42.9 | 41.3 | +4.0 [-1.2, +9.2] | 100 | met |
| morning_up_peak | liftzero_bc | full | avg_wait | 25.2 | 23.1 | +9.1 [-4.9, +26.7] | 100 | missed |
| morning_up_peak | liftzero_bc | full | p95_wait | 68.3 | 63.4 | +7.7 [-2.5, +20.0] | 100 | met |
| priority_passenger | liftzero_bc | full | avg_wait | 12.9 | 13.1 | -1.7 [-4.7, +1.4] | 100 | met |
| priority_passenger | liftzero_bc | full | p95_wait | 39.5 | 40.3 | -1.9 [-6.2, +2.6] | 100 | met |
| stress_scale | liftzero_bc | full | avg_wait | 17.6 | 17.2 | +2.2 [-0.6, +5.0] | 100 | met |
| stress_scale | liftzero_bc | full | p95_wait | 68.4 | 67.5 | +1.3 [-1.4, +4.1] | 100 | met |
| two_way | liftzero_bc | full | avg_wait | 14.9 | 14.5 | +2.6 [-0.8, +6.2] | 100 | met |
| two_way | liftzero_bc | full | p95_wait | 45.8 | 45.2 | +1.2 [-3.5, +6.6] | 100 | met |
| up_peak | liftzero_bc | full | avg_wait | 31.6 | 29.9 | +5.8 [-3.7, +17.1] | 100 | missed |
| up_peak | liftzero_bc | full | p95_wait | 79.7 | 77.8 | +2.3 [-5.1, +10.2] | 100 | met |
| car_breakdown | liftzero_bc_cnp | cnp_astar | avg_wait | 24.2 | 24.3 | -0.5 [-6.6, +5.8] | 100 | met |
| car_breakdown | liftzero_bc_cnp | cnp_astar | p95_wait | 65.3 | 65.2 | +0.2 [-5.3, +6.2] | 100 | met |
| demo_story | liftzero_bc_cnp | cnp_astar | avg_wait | 40.3 | 38.8 | +3.9 [-0.6, +8.4] | 100 | met |
| demo_story | liftzero_bc_cnp | cnp_astar | p95_wait | 100.2 | 99.5 | +0.7 [-3.8, +6.0] | 100 | met |
| down_peak | liftzero_bc_cnp | cnp_astar | avg_wait | 21.9 | 22.0 | -0.4 [-2.6, +1.9] | 100 | met |
| down_peak | liftzero_bc_cnp | cnp_astar | p95_wait | 62.8 | 62.7 | +0.1 [-2.4, +2.6] | 100 | met |
| evening_down_peak | liftzero_bc_cnp | cnp_astar | avg_wait | 19.8 | 20.1 | -1.5 [-3.7, +0.7] | 100 | met |
| evening_down_peak | liftzero_bc_cnp | cnp_astar | p95_wait | 58.4 | 59.8 | -2.3 [-5.1, +0.4] | 100 | met |
| fire_emergency | liftzero_bc_cnp | cnp_astar | avg_wait | 67.9 | 71.8 | -5.4 [-8.4, -3.0] | 100 | met |
| fire_emergency | liftzero_bc_cnp | cnp_astar | p95_wait | 261.7 | 269.2 | -2.8 [-4.5, -1.4] | 100 | met |
| interfloor | liftzero_bc_cnp | cnp_astar | avg_wait | 11.6 | 11.8 | -1.0 [-3.0, +1.2] | 100 | met |
| interfloor | liftzero_bc_cnp | cnp_astar | p95_wait | 34.0 | 34.9 | -2.6 [-6.7, +1.7] | 100 | met |
| interfloor_light | liftzero_bc_cnp | cnp_astar | avg_wait | 9.7 | 9.8 | -1.5 [-3.3, +0.2] | 100 | met |
| interfloor_light | liftzero_bc_cnp | cnp_astar | p95_wait | 25.5 | 26.8 | -4.9 [-8.6, -1.0] | 100 | met |
| lunch_two_way | liftzero_bc_cnp | cnp_astar | avg_wait | 14.0 | 13.9 | +0.5 [-1.7, +2.8] | 100 | met |
| lunch_two_way | liftzero_bc_cnp | cnp_astar | p95_wait | 44.0 | 42.4 | +3.7 [-0.2, +7.7] | 100 | met |
| morning_up_peak | liftzero_bc_cnp | cnp_astar | avg_wait | 39.1 | 38.5 | +1.5 [-5.3, +8.8] | 100 | met |
| morning_up_peak | liftzero_bc_cnp | cnp_astar | p95_wait | 93.2 | 94.4 | -1.3 [-6.7, +4.4] | 100 | met |
| priority_passenger | liftzero_bc_cnp | cnp_astar | avg_wait | 13.0 | 13.3 | -2.1 [-4.1, -0.2] | 100 | met |
| priority_passenger | liftzero_bc_cnp | cnp_astar | p95_wait | 39.5 | 40.5 | -2.5 [-5.9, +0.9] | 100 | met |
| stress_scale | liftzero_bc_cnp | cnp_astar | avg_wait | 22.9 | 23.1 | -0.7 [-2.2, +0.8] | 100 | met |
| stress_scale | liftzero_bc_cnp | cnp_astar | p95_wait | 74.5 | 75.6 | -1.4 [-3.6, +0.6] | 100 | met |
| two_way | liftzero_bc_cnp | cnp_astar | avg_wait | 14.7 | 14.5 | +1.5 [-0.9, +3.9] | 100 | met |
| two_way | liftzero_bc_cnp | cnp_astar | p95_wait | 45.7 | 44.9 | +1.8 [-1.6, +5.5] | 100 | met |
| up_peak | liftzero_bc_cnp | cnp_astar | avg_wait | 43.4 | 42.9 | +1.1 [-4.0, +6.4] | 100 | met |
| up_peak | liftzero_bc_cnp | cnp_astar | p95_wait | 101.8 | 100.7 | +1.1 [-2.7, +4.8] | 100 | met |
