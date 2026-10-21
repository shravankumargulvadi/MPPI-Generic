# Retest of c0f2fa7

This directory retains the follow-up validation of commits `64afd3b` and `c0f2fa7` on the same RTX 3060/CUDA 12.6.85 environment as the [initial review](../README.md). See [the retest report](../../Adaptive_RMPPI_RETEST.md) for results and limitations. The implementation and benchmark are unchanged by this retest; only evidence, diagnostics and documentation are added.

`targeted_tests.cu` imports the earlier three review reproducers and the original four GPU tests, then adds six checks for GPU residual-trigger rejection, zero-budget telemetry, final-pass telemetry, disabling statistics, automatic combined-kernel selection and excessive shared-memory rejection. It stays outside the default project test suite. These diagnostics are intended to expose remaining parity failures instead of hiding them.

## Reproduction

Use the same prepared image/dependencies described in the initial review. Run from the repository root:

```bash
retest_dir="$PWD/build/adaptive-retest-reproduce"
mkdir -p "$retest_dir"
cp doc/adaptive_rmppi_validation/retest_c0f2fa7/CMakeLists.txt "$retest_dir/"
docker run --rm --runtime=nvidia --gpus all --network none \
  --user "$(id -u):$(id -g)" --workdir /validation --entrypoint bash \
  -v "$PWD:/repo:ro" -v "$retest_dir:/validation" trustnav:local_planner -lc '
    cmake -S /validation -B /validation/build \
      -DMPPI_BUILD_TESTS=ON -DMPPI_BUILD_EXAMPLES=ON \
      -DMPPI_CUDA_ARCH_LIST=86 -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DCMAKE_CXX_STANDARD=17 -DCMAKE_CUDA_STANDARD=17 -DCMAKE_BUILD_TYPE=Release &&
    cmake --build /validation/build --target adaptive_retest_diagnostics \
      adaptive_rmppi_reference_test adaptive_rmppi_test adaptive_rmppi_benchmark \
      rmppi_test rmppi_kernel_tests ddp_tests generic_feedback_controller_test -j2
  '
```

Inside the same mounted container, retain individual exit statuses. Some parity checks remain expected to fail at the tested commit. Complete builds and sanitizer runs before timing the benchmark.

```bash
/validation/build/adaptive_retest_diagnostics
/validation/build/mppi/tests/feedback_controllers/adaptive_rmppi_reference_test
/validation/build/mppi/tests/controllers/adaptive_rmppi_test
compute-sanitizer --tool memcheck --error-exitcode 99 \
  /validation/build/adaptive_retest_diagnostics \
  --gtest_filter=AdaptiveRetest.*:AdaptiveReview.SecondRefreshPreservesFirstRefreshPrefix
compute-sanitizer --tool racecheck --racecheck-report analysis \
  --num-cuda-barriers 4096 --error-exitcode 99 \
  /validation/build/mppi/tests/controllers/adaptive_rmppi_test \
  --gtest_filter=AdaptiveRMPPI.DisabledMatchesStockRMPPIWithSameSeed
/validation/build/mppi/examples/adaptive_rmppi_benchmark \
  30 300 /validation/benchmark.csv 0.5
```

Analyze the retained CSV on the host:

```bash
python3 doc/adaptive_rmppi_validation/analyze.py \
  --directory doc/adaptive_rmppi_validation/retest_c0f2fa7
# Add --plot with Matplotlib installed to regenerate comparison.png.
```

Log copies normalize trailing whitespace only. The aggregate figure uses the submitted proxy metrics; it is not a recovery trajectory plot or evidence of the missing nonlinear/mismatched-plant experiment.
