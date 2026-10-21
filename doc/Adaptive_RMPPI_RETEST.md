# Adaptive RMPPI follow up GPU validation

**Retested implementation: `c0f2fa751d368f199e4338ba9e79d77edab52df5`. Recommendation remains request changes.** The policy-prefix, telemetry, trigger-validation and kernel-selection fixes pass their targeted checks. Disabled/unreachable-radius parity and stock-versus-stock reproducibility still fail, racecheck still reports shared-memory hazards, and the submitted benchmark still does not measure the plan's nonlinear recovery criterion.

This retest covers commits `64afd3b` and `c0f2fa7`. Controller and benchmark code were not changed during retesting. The [earlier report](Adaptive_RMPPI_RESULTS.md) and its artifacts remain historical evidence for `fd97da2`; current evidence is in [retest_c0f2fa7](adaptive_rmppi_validation/retest_c0f2fa7/README.md).

## Fixes verified

| Check | Current result |
| --- | --- |
| Preserve the first refresh's policy prefix after a second refresh | passed; CPU and GPU both apply -0.2 at step 1, previously GPU applied 0 |
| Reject enabled GPU MODEL_RESIDUAL without changing existing config | passed; disabled residual configuration remains accepted |
| Report violations and overflows when refresh budget is zero | passed; one pass, zero refreshes, 256 violations and 256 overflows |
| Report unmet demand on the final replay | passed; refresh at step 1, overflow at step 2 |
| Clear last-iteration statistics when adaptation is disabled | passed |
| Select the combined kernel when enabled, even if split was selected | passed |
| Reject a combined-kernel configuration exceeding shared memory | passed |
| CPU switch from distance to model-residual triggering builds Jacobians | passed in the 14-test CPU reference suite |

The first seven checks also pass under CUDA memcheck: **7/7 passed, ERROR SUMMARY: 0 errors, exit 0**. This verifies the exercised fixes; it does not establish full GPU/CPU agreement for every trajectory, cost and importance-sampling correction.

## Regression and remaining failures

| Suite or diagnostic | Result |
| --- | --- |
| CPU reference | 14/14 passed |
| Submitted adaptive GPU suite | 2/4 passed; same two parity tests still fail |
| Existing RMPPI controller | 21/21 passed |
| Existing RMPPI kernels | 5/5 passed |
| Existing DDP | 10/10 passed |
| Existing generic feedback | 2/2 passed |
| Expanded isolated diagnostic executable | 9/13 passed; four parity/reproducibility assertions fail |
| CUDA racecheck on disabled parity | exit 99; 100 hazards displayed, 111 errors, 0 warnings |

The expanded executable includes the original four GPU tests, three earlier review diagnostics, and six new fix checks. Its counts overlap the standalone suite and are not additional independent coverage of every test. All 38 existing regressions pass.

Remaining failing tests in the expanded executable:

- `AdaptiveRMPPI.DisabledMatchesStockRMPPIWithSameSeed`
- `AdaptiveRMPPI.EnabledWithUnreachableRadiusMatchesStock`
- `AdaptiveReview.DisabledParityWithPinnedEvaluationKernels`
- `AdaptiveReview.StockMatchesStockWithPinnedKernels`

The stock-versus-stock failure still prevents attributing all parity differences specifically to adaptive feedback. The common rollout and init-eval kernel source was unchanged by the fixes. Racecheck, with `--num-cuda-barriers 4096`, still reports hazards involving both stock and adaptive feedback instantiations and init-eval. The retained run has no barrier-tracking overflow warning. The root cause of the parity differences remains unresolved; neither the failing assertions nor the racecheck reports should be relabeled as a clean baseline.

## Benchmark rerun

The unchanged submitted executable ran **30 paired seeds per severity, 180 trials, 54,000 cycles**, with arguments `30 300 benchmark.csv 0.5`. Settings remain horizon 50, 1024 rollouts, dt=0.02 s, one optimization iteration, lambda=4, alpha=0, threshold=10, Gaussian stddev=1, Q=diag(500,500,100,100), R=I, Q_f=I, W=I, one DDP iteration, and refresh budgets 2/rollout and 64/iteration with 64 slots. Seeds are 1000-1029; +x velocity kicks at cycle 50 are 0.5, 1 and 2 m/s. Both modes use the combined rollout kernel; evaluation kernels remain autotuned. Actuator limits are [-FLT_MAX, FLT_MAX].

Timing used the same RTX 3060 laptop, driver 595.99.02, CUDA toolkit 12.6.85, GCC 13.2.0, CMake 3.28.3, Release/C++17/sm_86 and default barrier configuration. Compilation and sanitizer runs finished before timing began. Normal laptop clock management and stock-before-adaptive ordering remain limitations. Exact implementation/environment identifiers are recorded in `environment.txt`.

The table retains the submitted proxy definitions. Survival is state cost below 1000 with finite states throughout, not recovery within a tolerance/deadline. Peak position compares next actual position against current nominal position, excludes velocity and includes the pre-disturbance interval. Latency is the **mean of per-trial p95s**, not a pooled cycle percentile. Refreshes/cycle is an average, not a refresh-load p95.

| Kick (m/s) | Mode | Survival | Mean peak position proxy (m) | Mean state cost | Mean trial p95 (ms) | Mean refreshes/cycle |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 | stock | 30/30 | 0.023078 | 888.357 | 0.517 | 0.000 |
| 0.5 | adaptive_v0 | 30/30 | 0.023077 | 888.374 | 1.154 | 8.403 |
| 1 | stock | 30/30 | 0.026047 | 922.357 | 0.521 | 0.000 |
| 1 | adaptive_v0 | 30/30 | 0.026045 | 922.327 | 1.197 | 8.494 |
| 2 | stock | 30/30 | 0.089520 | 898.852 | 0.525 | 0.000 |
| 2 | adaptive_v0 | 30/30 | 0.089373 | 899.032 | 1.189 | 8.404 |

| Kick (m/s) | Adaptive minus stock peak proxy mean [95% CI], m | Adaptive minus stock state cost mean [95% CI] | Added trial p95 mean [95% CI], ms | Reported overflow requests/cycle |
| --- | --- | --- | --- | --- |
| 0.5 | -5.67e-07 [-4.72e-06, 3.58e-06] | 0.0168 [-0.1042, 0.1472] | 0.6369 [0.6293, 0.6449] | 0.986 |
| 1 | -1.11e-06 [-3.29e-06, 8.74e-07] | -0.0297 [-0.2054, 0.1411] | 0.6759 [0.6675, 0.6844] | 1.110 |
| 2 | -0.000147 [-0.00044, 1e-06] | 0.1801 [-0.2237, 0.7785] | 0.6636 [0.6551, 0.6717] | 1.052 |

Intervals use 10,000 paired bootstrap resamples over seeds, RNG seed 20261021. All peak-proxy and state-cost difference intervals include zero. Both modes survive all 30 trials at every severity; each 30/30 survival estimate has a Wilson 95% interval of approximately 88.6%-100%. The degenerate all-success paired bootstrap interval does not prove equivalence.

Adaptive mode takes approximately **2.23-2.30 times** stock's mean trial p95 in this run: about 1.15-1.20 ms versus 0.52 ms. There is no statistically distinguishable improvement in the exported peak/cost proxies. Differences in absolute timing between the first review and this run are not a controlled before/after performance estimate.

Final-pass telemetry now exposes roughly 0.99-1.11 overflow requests per cycle on average; the first implementation reported zero in this benchmark. This is consistent with the targeted telemetry fix, and should not be interpreted as a measured increase in physical failures. These are request counts across passes, not the fraction of cycles hitting budget. The CSV still lacks that cycle-level fraction, refresh-load p95, maximum latency and deadline-miss count. A planning p95 below the 20 ms step interval does not establish zero deadline misses.

![Retest benchmark proxy comparison](adaptive_rmppi_validation/retest_c0f2fa7/comparison.png)

## Open acceptance items

The earlier policy-prefix defect is closed for the deterministic two-refresh reproducer. GPU residual-trigger rejection, final-pass/zero-budget telemetry, stale-statistics clearing and combined-kernel validation are verified for the supplied cases. The newly added CPU trigger-switch regression also passes.

The disabled-path parity and concurrency findings remain open. Complete seeded GPU/CPU trajectories, costs and importance-sampling correction agreement are still not established. A passing prefix check alone is not the complete Gate B acceptance test.

The benchmark source did not change: it still has only a matched linear double integrator, survival/position proxy metrics, and no full nominal, nonlinear or mismatched-plant recovery experiment. It does not export recovery trajectories or cycle-level timing/refresh traces. The plan's true recovery rate, deadline misses, nonlinear hypothesis, recovery plot and required load statistics therefore remain unvalidated. GPU V1 remains deliberately unsupported; the DDP exit-on-failure and unproved adaptive free-energy guarantee limitations remain.
