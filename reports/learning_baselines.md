# LiftZero Learning Baselines Report

| regime | policy | backend | samples | avg_wait_mean | avg_wait_ci_low | avg_wait_ci_high | p95_wait_mean | long_wait_pct_mean | throughput_mean | energy_mean | wall_s_sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| down_peak | collective | twin | 5 | 184.64 | 148.16 | 219.01 | 685.39 | 50.63 | 0.13 | 322.90 | 0.17 |
| down_peak | cost_greedy | twin | 5 | 34.87 | 31.08 | 38.95 | 139.85 | 15.70 | 0.19 | 738.00 | 0.25 |
| down_peak | nearest | twin | 5 | 153.51 | 137.09 | 168.69 | 671.87 | 42.34 | 0.13 | 378.80 | 0.18 |
| down_peak | random | twin | 5 | 55.58 | 47.53 | 63.19 | 263.89 | 21.57 | 0.18 | 691.60 | 0.22 |
| interfloor | collective | twin | 5 | 35.94 | 32.84 | 39.02 | 126.71 | 19.28 | 0.10 | 360.50 | 0.16 |
| interfloor | cost_greedy | twin | 5 | 14.85 | 13.07 | 17.14 | 50.94 | 3.50 | 0.10 | 523.80 | 0.17 |
| interfloor | nearest | twin | 5 | 21.31 | 19.10 | 23.52 | 86.62 | 10.91 | 0.10 | 483.90 | 0.16 |
| interfloor | random | twin | 5 | 25.15 | 22.75 | 28.07 | 67.23 | 7.67 | 0.10 | 550.30 | 0.16 |
| two_way | collective | twin | 5 | 48.16 | 37.38 | 63.53 | 219.68 | 20.05 | 0.16 | 470.10 | 0.20 |
| two_way | cost_greedy | twin | 5 | 19.23 | 16.79 | 21.50 | 64.04 | 5.50 | 0.18 | 689.90 | 0.22 |
| two_way | nearest | twin | 5 | 25.14 | 20.71 | 29.38 | 92.62 | 10.61 | 0.18 | 628.90 | 0.20 |
| two_way | random | twin | 5 | 29.32 | 26.60 | 31.48 | 90.97 | 11.46 | 0.18 | 651.40 | 0.19 |
| up_peak | collective | twin | 5 | 102.85 | 65.11 | 137.44 | 228.43 | 57.48 | 0.16 | 348.60 | 0.11 |
| up_peak | cost_greedy | twin | 5 | 43.16 | 30.55 | 61.06 | 101.84 | 29.43 | 0.19 | 447.00 | 0.13 |
| up_peak | nearest | twin | 5 | 66.67 | 41.95 | 88.64 | 143.66 | 45.41 | 0.18 | 407.70 | 0.32 |
| up_peak | random | twin | 5 | 50.23 | 30.79 | 79.17 | 114.04 | 29.64 | 0.18 | 462.80 | 0.11 |

