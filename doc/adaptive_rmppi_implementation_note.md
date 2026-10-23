# Adaptive RMPPI: implementation note

Status: prototype; **no measured cart-pole recovery advantage after trigger and common-parameter tuning**. The [2,838-trial cart-pole study](Adaptive_RMPPI_CARTPOLE_STUDY.md) calibrates radii 4 and 4.5 and includes 1,620 frozen evaluation trials on seeds 5000-5029. Stock/adaptive primary recovery is identical: 61/360 disturbed trials with original settings and 254/360 with cart weight 200/force 20 N. Adaptive mean trial p95 remains about 1.7x stock on disturbed cases, and even a zero-refresh nominal control adds 0.082 ms [95% CI 0.066, 0.098]. Keep adaptive mode disabled for these tested settings. Earlier functional tests and [lineinfo racecheck findings](Adaptive_RMPPI_RETEST.md) are unchanged; this study edits no controller/kernel code.

## Where adaptation happens
- `AdaptiveDDPFeedback` (`include/mppi/feedback_controllers/adaptive/adaptive_ddp.cuh`) computes the baseline gains with the same DDP problem as `DDPFeedback`. It also owns a bank of refreshed policies (`MAX_POLICIES` slots) and one policy assignment per rollout (slot id + start timestep).
- `AdaptiveRobustMPPIController` (`include/mppi/controllers/R-MPPI/adaptive_robust_mppi_controller.cuh`) overrides the new `RobustMPPIController::launchRolloutKernels()` hook. With `AdaptiveFeedbackConfig::enabled == false` it calls the parent, so behaviour is stock RMPPI.
- `AdaptiveRMPPIReference` (`adaptive_rmppi_reference.h`) is the CPU oracle. It also provides the refresh solver used by the GPU path.

## Tracking law
`u = v + K~_t (x_t - x*_t)`, where `x*` is the paired nominal state (unchanged from stock). Only the gain `K~` changes. The recovery trajectory `x~` is used only to judge whether the gain is still valid. It is never used as the tracking target.

## Trigger
- V0 (CPU and GPU): `sqrt(sum_i w_i (x_i - x~_i)^2) > distance_radius`, with optional angle wrapping.
- V1 (CPU reference only): one-step residual between the nonlinear step and the cached affine model, `x~_{t} + A_d (x_{t-1} - x~_{t-1}) + B_d (u_{t-1} - u~_{t-1})`, plus a hard distance radius. Here `A_d = I + A dt` and `B_d = B dt`, matching the DDP solver's Euler discretization.

## Refresh
On a trigger at timestep t, DDP is re-solved over the remaining horizon `T - t`, starting from the actual state. It uses the same Q, R, Q_f and control limits, and it tracks the nominal reference suffix (RMPPI's nominal trajectory and controls). A refresh is skipped when fewer than 2 steps remain. Refreshes are capped per rollout and per optimization iteration, and the GPU path is also capped by the number of bank slots. Overflows, solver failures (non-finite output) and short-horizon skips are counted and returned in `AdaptiveRolloutStats`. The fallback is to keep the previous gains.

## Staged GPU execution (per optimization iteration)
1. Save the sampled controls. The rollout kernel overwrites them with the applied controls.
2. Pass A: the combined `rolloutRMPPIKernel` runs. `DeviceAdaptiveDDP::k` records each rollout's first V0 violation (t and state).
3. The host refreshes the recorded rollouts in sample order, within budget.
4. Pass B: the noise is restored and all rollouts are re-run from t = 0 with their new policies. All costs, likelihood-ratio terms and feedback-cost terms are recomputed from the trajectory that was actually simulated, so old and new trajectories are never mixed. Steps 2-4 repeat until no policy changes or `max_refreshes_per_rollout` is reached. The final pass also records violations; it reports unmet refresh requests as budget overflows without attempting another solve. Refreshed policy slots preserve the earlier gain/waypoint prefix, so replay retains preceding refreshes.
- The GPU feedback rejects an enabled `MODEL_RESIDUAL` config with `std::invalid_argument` (V1 is CPU-reference only).
- Adaptive mode requires the single combined kernel, because the policy lookup is keyed by sample index. It selects that kernel automatically (so the inherited sampled-trajectory paths stay consistent) and throws `std::runtime_error` if the combined kernel does not fit in shared memory, rather than falling back to split kernels.
- Re-running every rollout, not just the affected ones, is simpler and always consistent. It costs one extra full kernel per refresh round.

## Differences between the CPU oracle and the GPU path
- The oracle refreshes immediately inside a sequential rollout. The GPU refreshes between passes. Both produce the same (sample, t, gain) for the first refresh of each rollout. When the per-iteration budget binds, the GPU allocates refreshes by sample index within a pass, while the oracle allocates them in rollout order. Results can then differ.
- V1 is not implemented on the GPU. That would need cached `A_d`/`B_d` in the bank.

## Theoretical limitations
- With state-dependent, per-rollout gains, the actual-system control distribution is no longer the fixed-gain feedback policy that the stock RMPPI change-of-measure assumes. The kernel still adds the same likelihood-ratio and feedback-cost terms, computed from the feedback actually applied. Whether the RMPPI free-energy growth bound still holds has **not** been shown.
- The nominal-state selection, cost function and sampling policy are unchanged.
- DDP calls `std::exit` on an LDLT failure inside the backward pass. The "solver failure" fallback only catches non-finite output.

## Combined-kernel nominal-cost race (stock RMPPI)

`rolloutRMPPIKernel` (the combined kernel, one sample per `threadIdx.x`) combined the nominal cost with the actual
rollout's tracking cost using an index copied from the split cost kernel, where each block holds a single sample. As a
result, every actual-state thread in a block read-modified-wrote the nominal cost of the block's first sample, while
the other nominal samples never received the value-function mix or their likelihood-ratio term. The update now uses
`thread_idx + blockDim.x * blockDim.y * NOMINAL_STATE_IDX`, which matches the split kernel and the CPU reference in
`tests/include/kernel_tests/core/rmppi_kernel_test.cu`. This is a stock-path change. It is the most likely cause of
the stock-vs-stock same-seed divergence and racecheck reports in the retest, but this has not been confirmed on a GPU.

## Recovery benchmark (nonlinear)

`examples/adaptive_rmppi_recovery_benchmark` runs a cartpole balancing task with three scenarios: `nominal` (no
disturbance), `kick` (pole angular-velocity kicks of 1, 2 and 3 rad/s at cycle 50) and `mismatch` (the same kicks,
with a simulated plant whose pole mass is 1.5 kg and pole length 1.3 m; the planner model is unchanged). Seeds are
1000+ and mode order alternates by seed. Recovery success, the failure checks and the error weights are defined in
the file header and were fixed before any run; the trigger radius has not been tuned. Per-trial rows include recovery
time, peak error, post-disturbance cost, trial p95/max latency, deadline misses (latency > 20 ms), refresh mean/p95
and the fraction of cycles hitting the budget. An optional per-cycle trace is also written.
`examples/adaptive_rmppi_recovery_analyze.py` produces paired-bootstrap and Wilson intervals, `table.md`,
`summary.json` and, with `--plot`, a recovery plot plus a latency histogram. The benchmark was run on an RTX 3060 for 30 paired seeds per condition; results and raw traces are in [the latest retest artifacts](adaptive_rmppi_validation/retest-6731dcb/README.md). Neither mode recovered in the kicked conditions, and adaptive mode used its 64-refresh iteration budget on 99.8-100% of cycles. The chosen kicks therefore need calibration to include recoverable regimes before this can establish whether adaptation helps.

```bash
./examples/adaptive_rmppi_recovery_benchmark 30 300 recovery.csv 0.5 recovery_trace.csv
python3 ../examples/adaptive_rmppi_recovery_analyze.py recovery.csv --trace recovery_trace.csv --plot
```

## Strided cost-reduction race (stock MPPI/RMPPI combined kernels)

`costArrayReduction` and `multiCostArrayReduction` finish with a warp-synchronous tail (`warpReduceAdd`, no
barriers) once 32 or fewer partial sums remain. The combined kernels call them with `stride = blockDim.x` to reduce
across `threadIdx.y` for each sample. Those partial sums belong to threads in different warps, so the tail read and
wrote shared memory across warps without synchronization whenever `blockDim.y` is a power of two of at least 4. Both
the adaptive tests and the benchmarks use `(64, 4, 2)`. Strided reductions now use `__syncthreads` at every level, and
contiguous (`stride == 1`) reductions are unchanged. The isolated kernel test uses `blockDim.y` 1-3, which never
reaches this path, so its remaining racecheck reports have a different source. A `-lineinfo` racecheck run is needed
to locate it.

Recovery benchmark calibration: arguments 6 and 7 set the seed offset and the kick list, so kicks and the trigger
radius can be chosen on seeds below 1000 and then frozen, e.g.
`adaptive_rmppi_recovery_benchmark 10 300 cal.csv 2.0 - 0 0.25,0.5,0.75`.

Plant-side feedback: before this fix, both benchmarks applied only `getControlSeq().col(0)` to the simulated plant.
RMPPI optimizes from a nominal state x* that can differ from the plant state, so the real system also needs the
tracking term. Both benchmarks now apply `u*_0 + K_0 (x - x*_0)`, using `getFeedbackControl` against
`getTargetStateSeq().col(0)`. Results recorded for `6731dcb` and earlier were produced without this term.
