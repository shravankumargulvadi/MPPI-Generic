# Retest of MR commit 6731dcb

Latest tested MR head: `6731dcbf08cf4258af83c6bf605eaa69169cae86`.

## Build and test environment

- NVIDIA GeForce RTX 3060 Laptop GPU, 6 GiB
- Driver 595.99.02; CUDA container toolkit 12.6.85
- `trustnav:local_planner` container; GCC 13.2.0; CMake 3.28.3
- Release, C++17, CUDA architecture 86
- Build included the adaptive tests, prior review diagnostics, existing RMPPI/DDP tests, and both RMPPI benchmarks.

## Commands

The tests were run from a writable mounted directory in the CUDA container:

```bash
adaptive_rmppi_reference_test
adaptive_rmppi_test
adaptive_retest_diagnostics
rmppi_kernel_tests
rmppi_test
ddp_tests
generic_feedback_controller_test
```

Results: CPU reference 14/14, submitted adaptive GPU 4/4, expanded review diagnostics 13/13, kernel tests 5/5, RMPPI controller 21/21, DDP 10/10, and generic feedback 2/2 passed. The expanded diagnostics include the same four adaptive tests, so counts overlap. CUDA memcheck on the four adaptive GPU tests passed with zero errors.

Racecheck on `AdaptiveReview.StockMatchesStockWithPinnedKernels` passed its numerical assertion but reported 65 shared-memory hazards. The isolated `RMPPIKernels.ValidateCombinedRMPPIRolloutKernelAgainstCPU` test passed numerically but racecheck reported 10 hazards. Both runs used `--num-cuda-barriers 4096`; therefore racecheck remains unresolved.

## Recovery benchmark

Reproduce the 30 paired seeds and analyzer output with:

```bash
adaptive_rmppi_recovery_benchmark 30 300 recovery.csv 0.5 recovery_trace.csv
python3 examples/adaptive_rmppi_recovery_analyze.py recovery.csv --trace recovery_trace.csv --out analysis --plot
```

The benchmark generated 420 trial rows (30 seeds x 7 conditions x 2 modes) and per-cycle traces. Its strict recovery definition yielded 30/30 successes for both modes in nominal balancing and 0/30 for both modes in every kick and plant-mismatch condition. For kick magnitudes 2 and 3 rad/s, and for all mismatch magnitudes, all trials hit a safety constraint. Kick magnitude 1 rad/s did not hit the safety constraint but still had no trial meet the recovery definition.

Adaptive mode refreshed about 61.5-63.2 rollouts per cycle and hit its 64-refresh budget on 99.8-100% of cycles. Mean trial p95 latency was 1.55-1.77 ms versus 0.48-0.49 ms for stock, an added 1.06-1.28 ms. No trial exceeded the 20 ms cycle deadline. Peak-error and cost paired-bootstrap intervals include zero; recovery success was identical. The mismatch-severity-1 cost estimate has very wide uncertainty because both modes often fail early, so it should not be read as a stable cost comparison.

This is a useful nonlinear stress test but it does not show a recovery benefit: every kicked condition is outside a useful success range for both controllers, and the adaptive controller runs at its refresh cap even in nominal balancing. A calibrated disturbance sweep with at least one recoverable and one challenging regime is needed before concluding whether V0 improves recovery.

`recovery.png`, `table.md`, and `summary.json` were produced by the supplied analyzer. The plotting package was missing in the CUDA image, so plotting was run in the host plotting environment after the GPU run. The analyzer's success-rate Wilson interval prints a tiny floating-point negative lower bound as `-0.00` for zero successes; the underlying bound is effectively zero.

See `recovery.csv`, `recovery_trace.csv`, and the adjacent test and sanitizer logs for raw evidence.
