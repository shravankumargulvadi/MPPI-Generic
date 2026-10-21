## 2026-10-21 — Adaptive RMPPI (doc/Adaptive_RMPPI_Implementation_Plan.docx)

### Milestone 1: CPU reference + feedback policy bank + adaptive controller

**Status:** In Progress

**Changed:**
- Added the CPU oracle (V0/V1 triggers, budgets, fallback), AdaptiveDDPFeedback, and AdaptiveRobustMPPIController (staged passes A/B).
- Added a behaviour-preserving `RobustMPPIController::launchRolloutKernels()` hook.

**Key decisions:**
- Pass B re-runs all rollouts so that costs always match the simulated trajectories. Adaptive mode forces the single kernel.

**Assumptions:**
- The refresh tracks RMPPI's nominal trajectory suffix. The gains are validated against the DDP linearization trajectory.

**Validation:**
- CUDA 12.4.1 container: adaptive_rmppi_reference_test passes (13/13). adaptive_rmppi_test, rmppi_test and adaptive_rmppi_benchmark compile; none were run because there is no GPU.

**Next step:**
- Run adaptive_rmppi_test and the Phase 4 benchmark on a GPU machine.

### Milestone: Address Devin Review findings on PR #1

**Status:** Done

**Changed:**
- Adaptive mode now validates/selects the combined kernel (throws if it does not fit shared memory); disabled iterations clear stats.
- Repeated refreshes carry earlier policy segments forward; last pass records violations as budget overflows; residual-trigger switch rebuilds baseline Jacobians.

**Validation:**
- CPU reference tests 14/14 in CUDA container; GPU targets compile only (no GPU).

**Next step:**
- Run `adaptive_rmppi_test` and the benchmark on a GPU machine.

### Milestone: Respond to retest of c0f2fa7

**Status:** Done (compile-only)

**Changed:**
- Fixed combined RMPPI rollout kernel nominal-cost indexing (cross-thread race; stock path).
- Added cartpole recovery benchmark (nominal/kick/mismatch) and paired-seed analyzer.

**Validation:**
- CPU reference 14/14; GPU targets, rmppi_kernel_tests and the new benchmark compile only (no GPU); analyzer smoke-tested on synthetic CSV.

**Next step:**
- On GPU: rerun stock-vs-stock and adaptive parity, racecheck, rmppi_kernel_tests, then the recovery benchmark.
