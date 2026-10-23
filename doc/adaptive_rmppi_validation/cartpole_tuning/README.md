# Cart-pole calibration and experiment artifacts

See [the report](../../Adaptive_RMPPI_CARTPOLE_STUDY.md) and [registered protocol](PROTOCOL.md). Controller behavior is from `7c54a0f`, on branch parent `26d7221`; only the cart-pole benchmark and analysis are changed. The first coarse radius scan used the unmodified benchmark binary; subsequent runs used the extended harness with explicit settings. `source_and_binary_sha256.txt` records the extended source and all three executable hashes.

## Trial inventory

| Stage | Seeds | Trials |
| --- | --- | ---: |
| Coarse radius scan | 0-2 | 144 |
| Refined radius scan | 10-14 | 150 |
| Common parameter screen | 20-22 | 420 |
| Cart-cost follow-up | 100-102 | 378 |
| Radius check for selected stronger setting | 200-204 | 120 |
| Default compatibility and diagnostic smoke | 0 | 6 |
| Frozen evaluation | 5000-5029 | 1620 |
| Total | | 2838 |

Final evaluation includes 780 trials in each of two shared settings, plus 60 nominal-only no-refresh-control trials. Both modes use the same seed and common physical/controller parameters; mode order alternates by seed. No final evaluation result influenced calibration or configuration selection. The primary criterion is the original weighted full-state error <0.2 by 2 seconds after the kick and maintained through the episode, with safety constraints. Pole-only success is a secondary angle/rate metric with the same weights/tolerance/deadline, maintained through the recorded segment; cart-limit termination censors later pole behavior.

## Build and rerun

The offline validation image contains CUDA, Eigen, googletest and the repository dependencies. Use this folder's CMake wrapper as the source tree, mount the repository at `/repo`, and build in a writable directory:

```bash
cmake -S /repo/doc/adaptive_rmppi_validation/cartpole_tuning -B /validation/build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_CUDA_FLAGS=-lineinfo -DMPPI_BUILD_EXAMPLES=ON -DMPPI_BUILD_TESTS=ON
cmake --build /validation/build --target adaptive_rmppi_recovery_benchmark adaptive_cartpole_h100 adaptive_cartpole_h150 -j2
python3 /repo/doc/adaptive_rmppi_validation/cartpole_tuning/run_calibration.py radius --out /validation/cartpole-tuning/refined-radius
python3 /repo/doc/adaptive_rmppi_validation/cartpole_tuning/run_calibration.py screen --radius 4 --out /validation/cartpole-tuning/parameter-screen
python3 /repo/doc/adaptive_rmppi_validation/cartpole_tuning/run_calibration.py followup --radius 4 --out /validation/cartpole-tuning/cart-cost-followup
python3 /repo/doc/adaptive_rmppi_validation/cartpole_tuning/run_calibration.py evaluation --out /validation/cartpole-tuning/evaluation
```

The runner's `--build` option changes the build directory. Initial and final-radius-check settings are recorded in the adjacent manifests and protocol. The initial scan ran radii 0.5,1,2,4,8,16,32,1e6; the stronger-setting check ran radii 4,4.5,5,6 with `--cart-weight=200 --force=20 --diagnostics`.

The extended benchmark preserves positional arguments and defaults, and adds optional flags after the kick list:

```bash
adaptive_rmppi_recovery_benchmark 30 300 results.csv 4.5 trace.csv 5000 \
  0.25,0.5,0.75,1,1.25,1.5 --cart-weight=200 --force=20 --diagnostics
```

Options: `--force`, `--cart-weight`, `--noise`, `--lambda`, `--ddp-iterations`, `--budget` (0-64), `--modes=both|stock|adaptive`, and `--scenarios=all|nominal,kick,mismatch`. `--diagnostics` records relative suffix-gain changes and the total final actual-system importance weight belonging to refreshed samples. These diagnostics run after the planning timer; gain/weight columns are not measured when the flag is omitted. `ADAPTIVE_CARTPOLE_HORIZON` changes the compile-time horizon, default 50 steps. Study variants use 100 and 150 steps.

## Analysis

```bash
python3 analyze_study.py .
python3 analyze_study.py . --plot
```

NumPy is required; plotting additionally requires Matplotlib. The script validates CSV uniqueness/finite values and exact paired final seed sets; summarizes every development case; computes 10,000-resample paired bootstrap intervals with RNG seed 20261023; and compares compressed final control/state traces. It does not interpret an all-equal bootstrap interval as equivalence. Cumulative costs from safety-truncated failures are excluded from the complete-pair cost comparison.

`evaluation_summary.json` contains per-condition estimates and confidence intervals. `evaluation_table.md` is the compact table. `trace_comparison.json` records exact command matches at the serialized float precision and refreshed-sample weight mass. `evaluation/*-trace.csv.gz` holds the raw traces and can be read directly with Python `gzip.open(..., 'rt')`.

## Interpretation and verification

Radii 4 and 4.5 greatly reduce nominal refresh demand, but are empirical tuning choices. They do not certify the validity of local gains. Full-state recovery was identical between modes in every paired final seed/condition; generic cart-weight and force-limit changes improved both modes. The no-refresh control retains measurable adaptive bookkeeping overhead.

`validation.txt` records build/environment and smoke checks. All three horizon variants compiled. The extended default smoke reproduces original outcomes/costs/refresh metrics within the earlier CSV precision; telemetry schemas, finite values, actuator bounds, and weight bounds were checked. Production controller/feedback/kernel files were not edited. Earlier racecheck findings remain unresolved.
