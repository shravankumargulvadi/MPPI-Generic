# Adaptive RMPPI review evidence

For the current follow-up results and artifacts, see [retest_c0f2fa7](retest_c0f2fa7/README.md). This directory retains the original review evidence.

These artifacts validate implementation commit `fd97da2af63e3464d1f4e12288a68ecfbaef7cc9`. See [the review report](../Adaptive_RMPPI_RESULTS.md) for interpretation. No controller or benchmark source was changed during review. Retained log copies have trailing whitespace normalized; their diagnostic content is unchanged.

- `benchmark.csv`: 180 submitted-executable trials: 30 paired seeds, three severities, two modes, 300 cycles/trial. Its success and peak columns are the submitted proxy definitions, not the recovery metrics in the plan.
- `summary.json`, `table.md`, `comparison.png`: derived summaries, paired bootstrap uncertainty and aggregate proxy comparison. This is not a recovery trajectory plot.
- `analyze.py`: regenerate summaries with NumPy; add `--plot` when Matplotlib is installed. Bootstrap seed 20261021, 10,000 resamples, pairs matched by seed within severity.
- `cpu-reference.log`, `gpu-tests.log`: submitted CPU/GPU suites.
- `memcheck.log`: no invalid-access errors; exit 1 remains an assertion failure, not a successful test suite.
- `racecheck.log`: targeted disabled-parity test; racecheck exit 99, 116 errors/0 warnings, with a barrier-tracking limit of 4096. Both stock and adaptive feedback kernel instantiations and the common init-eval kernel appear. Causality and provenance of those hazards are unresolved.
- `review_diagnostics.cu`: isolated extra tests reproducing pinned-kernel disabled parity, lost policy prefix on a second refresh, and stock-versus-stock reproducibility. These tests intentionally fail at the reviewed commit; they are not registered in the project's default test suite.
- `replay-and-parity.log`, `stock-control.log`: added diagnostic outputs with default barrier options.
- `no-barriers.log`: supplemental build with `CMAKE_USE_CUDA_BARRIERS` undefined in the diagnostic target. All three added checks still fail; changing this option alone did not fix reproducibility.
- `regressions.log`: end-of-suite summaries for the existing GPU regression executables.
- `CMakeLists.txt`: isolated wrapper using an installed GoogleTest source tree to avoid a download during CMake configuration.

## Reproduction

The measured environment used the existing local `trustnav:local_planner` image with CUDA 12.6.85, GCC 13.2, CMake 3.28.3, Eigen3, yaml-cpp, zlib and GoogleTest sources at `/usr/src/googletest`. The image identifier is recorded in the report. An equivalent prepared CUDA development image can be substituted; it needs these dependencies. Docker GPU access on this host requires `--runtime=nvidia` as well as `--gpus all`.

Run from the repository root. The source mount is read-only; tests that save `.npy` data run in the writable output directory.

```bash
review_dir="$PWD/build/adaptive-review-reproduce"
mkdir -p "$review_dir"
cp doc/adaptive_rmppi_validation/CMakeLists.txt "$review_dir/"
cp doc/adaptive_rmppi_validation/review_diagnostics.cu "$review_dir/"
docker run --rm --runtime=nvidia --gpus all --network none \
  --user "$(id -u):$(id -g)" --workdir /validation --entrypoint bash \
  -v "$PWD:/repo:ro" -v "$review_dir:/validation" trustnav:local_planner -lc '
    cmake -S /validation -B /validation/build \
      -DMPPI_BUILD_TESTS=ON -DMPPI_BUILD_EXAMPLES=ON \
      -DMPPI_CUDA_ARCH_LIST=86 -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DCMAKE_CXX_STANDARD=17 -DCMAKE_CUDA_STANDARD=17 -DCMAKE_BUILD_TYPE=Release &&
    cmake --build /validation/build --target \
      adaptive_rmppi_reference_test adaptive_rmppi_test adaptive_rmppi_benchmark \
      rmppi_test rmppi_kernel_tests ddp_tests generic_feedback_controller_test \
      adaptive_review_diagnostics adaptive_review_no_barriers -j2
  '
```

Run tests inside the same mounted container/environment, retaining each exit status. At the reviewed commit, the adaptive suite and added diagnostics are expected to fail; do not put benchmark execution behind an `&&` chain that would skip it on these failures.

```bash
/validation/build/mppi/tests/feedback_controllers/adaptive_rmppi_reference_test
/validation/build/mppi/tests/controllers/adaptive_rmppi_test
/validation/build/adaptive_review_diagnostics --gtest_filter=AdaptiveReview.*
/validation/build/adaptive_review_no_barriers
compute-sanitizer --tool memcheck --error-exitcode 99 \
  /validation/build/mppi/tests/controllers/adaptive_rmppi_test
compute-sanitizer --tool racecheck --racecheck-report analysis \
  --num-cuda-barriers 4096 --error-exitcode 99 \
  /validation/build/mppi/tests/controllers/adaptive_rmppi_test \
  --gtest_filter=AdaptiveRMPPI.DisabledMatchesStockRMPPIWithSameSeed
/validation/build/mppi/examples/adaptive_rmppi_benchmark \
  30 300 /validation/benchmark.csv 0.5
```

For isolated timing, finish compilation and sanitizer runs before starting the benchmark. The reported run uses the submitted benchmark and default barrier options, not the supplemental no-barrier target. Analyze the retained evidence on the host:

```bash
python3 doc/adaptive_rmppi_validation/analyze.py
# Optional figure regeneration:
python3 doc/adaptive_rmppi_validation/analyze.py --plot
```
