# Dubins heading-kick validation

See [the report](../../Adaptive_RMPPI_DUBINS_RESULTS.md) and [protocol](PROTOCOL.md). The fetched MR head is `54895c8`; the new controller feature is from `f53c75f`. This study covers a changed plant-side execution policy, not only a new scenario for the older cart-pole implementation.

## Fixed experiment cases

Each case has 30 paired seeds 1000-1029, nominal plus heading kicks 1,2,2.5,3 rad, 300 cycles, radius 0.5, horizon 50, 1024 rollouts, dt=0.02, lambda=1, one MPPI and one DDP iteration. Speed is constrained to [0.5,2], yaw rate to [-2,2]. Sampling standard deviations are 0.5 (speed) and 1 (yaw rate). State cost weights and feedback Q are [0,10,2], Qf=Q, R=I, no executed-control cost. The recovery definition remains e=sqrt(y^2+wrap(yaw)^2)<0.2 within 4 seconds after the kick and maintained until episode end, with |y|<=10. No recovery threshold was retuned.

| Case | Plant recovery solve | Rollout refresh budget | Stock action |
| --- | --- | ---: | --- |
| submitted | on | 64 | stock RMPPI |
| rollout-only | off | 64 | stock RMPPI |
| plant-only | on | 0 | stock RMPPI |
| neither | off | 0 | stock RMPPI |
| reuse-ddp-baseline | on | 0 | outside the same radius, reuse the already computed baseline DDP first action |

The `submitted` case runs an exact source copy of the fetched benchmark, preserved as `submitted_heading.cu`. Other cases use optional review flags added to the benchmark; their defaults match the submitted behavior. The reuse baseline does not execute an extra DDP solve. All changes are in the review harness, diagnostics and documentation; the fetched controller/kernel implementation is unchanged.

Total timed benchmark trials: 1500 (300 per case). Additional default-compatibility/flag smoke trials: 12. The standalone mechanism probe is recorded separately.

## Reproduction

Mount the repository at `/repo` and a writable directory at `/validation` in the recorded CUDA container. The included CMake wrapper uses installed googletest to avoid downloads:

```bash
cmake -S /repo/doc/adaptive_rmppi_validation/dubins_heading -B /validation/build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_CUDA_FLAGS=-lineinfo -DMPPI_BUILD_TESTS=ON -DMPPI_BUILD_EXAMPLES=ON
cmake --build /validation/build --target adaptive_rmppi_heading_benchmark submitted_heading_benchmark dubins_recovery_probe adaptive_retest_diagnostics adaptive_rmppi_test adaptive_rmppi_reference_test rmppi_kernel_tests rmppi_test ddp_tests generic_feedback_controller_test -j2
python3 /repo/doc/adaptive_rmppi_validation/dubins_heading/run_cases.py --build /validation/build --out /validation/dubins-heading/cases
python3 /repo/doc/adaptive_rmppi_validation/dubins_heading/analyze.py /validation/dubins-heading --plot
```

The analysis requires NumPy and, for plotting, Matplotlib. Cases/commands/seeds are saved in `cases/manifest.json`. The new harness switches are `--plant-feedback=0|1`, `--rollout-budget=0..64`, and `--stock-ddp-action=0|1`, placed after the positional kick list. All defaults preserve the fetched benchmark. Trace columns `vel` and `yaw_rate` are requested controls before clamping; added `applied_vel`/`applied_yaw_rate` record constrained controls. The exact submitted trace lacks these extra fields, so its applied controls are reconstructed using the fixed actuator bounds.

The mechanism probe includes the benchmark setup and compares the existing baseline DDP trajectory/start, first gain and first action against `solveRecovery` for a 2.5 rad deviation. It then checks the action returned by `getFeedbackControl`, calls it twice, and verifies the disabled-plant path matches stock feedback. `recovery-probe.log` records all values.

## Statistical interpretation

`summary.json` contains per-condition paired bootstrap intervals and exact paired binary tests. `aggregate_summary.json` uses 10,000 bootstrap resamples over the 30 seed blocks (each block contains all four kicks), preserving dependence across repeated conditions. RNG seed is 20261023. Costs include every trial here because no safety constraint terminates an episode. Recovery-time comparisons use only pairs where both modes succeed; -1 in the compact table means no such pair exists. All-equal bootstrap intervals are not equivalence proofs.

`trace_summary.json` verifies the recovery norm and command bounds and compares paired clipped-control sequences. Reusing the baseline DDP action matches adaptive plant-only commands in all 150 paired trials. The default submitted trace has six-digit precision; the instrumented traces preserve round-trip float precision, so comparisons use a 1e-5 control tolerance where appropriate.

## Files

- `cases/*.csv`, `cases/*-trace.csv.gz`, logs and manifest: raw data.
- `summary.json`, `aggregate_summary.json`, `table.md`, `trace_summary.json`, `heading_study.png`: analysis.
- `submitted_heading.cu`, `recovery_probe.cu`, `run_cases.py`, `analyze.py`, `CMakeLists.txt`: reproducible sources.
- Regression, memcheck, rollout/parity/init-eval racecheck logs: correctness and remaining synchronization evidence.
- `source_and_binary_sha256.txt`, `validation.txt`: exact source/binary hashes and environment.

The rollout synchronization change clears the isolated rollout racecheck reproducer. The isolated init-eval run still reports sampling/output handoff errors and warp-reduction warnings, so the controller as a whole cannot be called racecheck-clean.
