# GPU retest of MR head 7c54a0f

Retested commits `7d073bf` (strided reduction synchronization and calibration arguments) and `7c54a0f` (apply feedback to the simulated plant in both benchmarks), on top of the prior MR head.

## Build and tests

Built in the CUDA container as Release, C++17, `sm_86`, with `-lineinfo`. Both benchmark binaries, adaptive tests, targeted diagnostics and existing regression targets compiled successfully.

| Suite | Result |
| --- | ---: |
| Adaptive CPU reference | 14/14 passed |
| Adaptive GPU | 4/4 passed |
| Expanded parity/fix diagnostics | 13/13 passed; includes the four adaptive GPU tests |
| Existing RMPPI kernel tests | 5/5 passed |
| Existing RMPPI controller tests | 21/21 passed |
| Existing DDP tests | 10/10 passed |
| Generic feedback controller | 2/2 passed |
| Adaptive GPU tests under CUDA memcheck | 4/4 passed; 0 memory errors |

The four adaptive GPU tests now pass disabled-mode parity and unreachable-radius parity. The separate diagnostics also report exact stock-vs-stock and stock-vs-disabled-adaptive controls for the tested cycles.

## Lineinfo racecheck

Racecheck used `--num-cuda-barriers 4096 --print-limit 30`. The isolated combined rollout test passes its numerical assertion but still reports **9 shared-memory hazards**. The stock parity and disabled adaptive parity diagnostics pass numerically but report **22 hazards** together. Reported source locations include `sampling_distribution.cu:193`, `rmppi_kernels.cu:768`, `dynamics.cu:124/163`, and `sampling_distribution.cu:302`; the retained reports do not point to either modified reduction helper. The strided reduction change therefore has not made the broader racecheck run clean, and the remaining synchronization reports need review.

## Feedback-corrected recovery benchmark

Reproduce the run with:

```bash
adaptive_rmppi_recovery_benchmark 30 300 recovery.csv 0.5 recovery_trace.csv
python3 examples/adaptive_rmppi_recovery_analyze.py recovery.csv --trace recovery_trace.csv --out analysis
python3 pole_only_summary.py recovery_trace.csv pole_only_summary.csv
```

The run contains 420 trials: 30 paired seeds for nominal balancing, three matched-plant kicks, and three mismatched-plant kicks, in stock and adaptive modes. Both modes apply `u*_0 + K_0(x - x*_0)` to the simulated plant. Seeds are 1000-1029. The original primary success criterion is unchanged: combined cart/pole weighted error below 0.2 within 2 seconds and maintained through the trial, while also remaining within the horizontal-angle and track limits.

Both modes succeed in 30/30 nominal trials and 0/30 trials for every kick/mismatch case under that combined criterion. For the 1 rad/s matched kick and 1 rad/s mismatched kick, neither mode hits a safety limit, and the pole-only angle/rate metric recovers in 30/30 trials. Mean pole-only recovery times are 0.367 s stock / 0.370 s adaptive for the matched plant, and 1.183 s / 1.177 s for the mismatched plant. The primary metric still fails because cart recovery is part of that definition. At 2 and 3 rad/s, neither mode meets the pole-only criterion; the trials also hit a safety constraint.

The pole-only secondary statistic is calculated by `pole_only_summary.py` from the trace. It uses `sqrt(w_theta * wrap(theta-pi)^2 + w_theta_dot * theta_dot^2) < 0.2`, with weights 1 and 0.1, reached within 2 seconds and maintained for the remaining recorded trace. It omits cart-position and track-limit criteria. It supplements, and does not replace, the primary benchmark metric.

Adaptive mode averages 61.2-63.2 refreshes/cycle and hits the 64-refresh budget on 99.2-100% of cycles, including nominal trials. Mean trial p95 is 1.64-2.33 ms adaptive versus 0.48-0.61 ms stock. There are no measured 20 ms deadline misses; the largest observed cycle latency is 13.924 ms. Paired peak-error and cost differences are near zero; where the bootstrap excludes zero for peak error, the absolute differences are under 0.0001. The 1 rad/s trials show that the original all-or-nothing result concealed pole recovery while cart position remained outside tolerance.

## Feedback-corrected linear benchmark

The second benchmark ran 30 paired seeds at each of three velocity-kick severities (180 trials total). Both modes reported 30/30 survival at each severity, with identical mean peak tracking error and cumulative cost at the displayed precision. Mean trial p95 latency was 0.577/1.307 ms (stock/adaptive) at severity 0.5, 0.732/1.642 ms at 1.0, and 0.801/1.788 ms at 2.0. Adaptive mode averaged 8.38-8.51 refreshes per cycle and about 0.97-1.05 overflow requests per cycle. This linear benchmark still does not test the nonlinear recovery hypothesis.

## Raw files

- `recovery.csv`, `recovery_trace.csv`, `table.md`, `summary.json`, `benchmark.log`, and `analysis.log`
- `linear.csv` and `linear-benchmark.log`
- Test logs and both lineinfo racecheck logs
- `pole_only_summary.py` and its output `pole_only_summary.csv`
- `environment.txt`

Results for `6731dcb` and earlier omit plant-side tracking feedback and are superseded for benchmark claims. The earlier GPU tests remain relevant to code paths unchanged by these commits, but their benchmark results should not be compared as controlled before/after timing.
