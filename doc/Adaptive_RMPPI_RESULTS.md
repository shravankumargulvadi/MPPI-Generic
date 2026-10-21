# Adaptive RMPPI results

**Status: not yet run.** The development VM has no GPU. Gate B (GPU vs CPU reference) and the Phase 4 closed-loop benchmark have not been executed, so this document makes no claim of improvement.

## What has been run
| Check | Where | Result |
|---|---|---|
| CPU reference tests (`adaptive_rmppi_reference_test`, 13 tests) | CUDA 12.4.1 container, no GPU | pass |
| GPU tests (`adaptive_rmppi_test`) | compiled only | not run |
| Existing `rmppi_test`, `ddp_tests` | compiled only (after refactor) | not run |
| `examples/adaptive_rmppi_benchmark` | compiled only | not run |

## To fill in once a GPU is available
- GPU model, driver, CUDA version, git SHA, and all controller settings (horizon, rollouts, lambda, alpha, dt, limits, Q/R/Q_f, trigger thresholds and budgets).
- At least 30 paired seeds per scenario and severity, comparing stock RMPPI, adaptive V0, and adaptive V1 (V1 currently runs only in the CPU reference).
- Metrics, each with a paired-difference CI: recovery success rate, peak tracking error, closed-loop cumulative cost, control-cycle latency p95, and refreshes/overflows per cycle.
- One recovery plot and one latency histogram.
- Thresholds must be tuned on seeds disjoint from the evaluation seeds.

## How to run
```
cmake -DMPPI_BUILD_TESTS=ON -DMPPI_BUILD_EXAMPLES=ON .. && make adaptive_rmppi_test adaptive_rmppi_benchmark
./tests/controllers/adaptive_rmppi_test
./examples/adaptive_rmppi_benchmark 30 300 adaptive_rmppi_benchmark.csv 0.5   # seeds steps out radius
```
The benchmark kicks the double-integrator velocity by 0.5, 1 or 2 m/s at step 50. It writes one CSV row per (severity, mode, seed) with these columns: success, peak tracking error, cumulative cost, p95 latency, violations, refreshes and overflows.
