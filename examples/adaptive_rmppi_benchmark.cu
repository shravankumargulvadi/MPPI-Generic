/**
 * Paired closed-loop disturbance-recovery benchmark: stock RMPPI vs adaptive RMPPI (V0 distance trigger).
 * Plant: double integrator with DoubleIntegratorCircleCost. At step `disturbance_step` the true plant velocity is
 * kicked by `severity` m/s along +x. Every mode sees the same seed, plant, cost, horizon, rollouts, lambda, limits.
 *
 * Usage: adaptive_rmppi_benchmark [num_seeds=30] [steps=300] [out.csv=adaptive_rmppi_benchmark.csv] [radius=0.5]
 * One CSV row per (severity, mode, seed). Requires a CUDA GPU.
 */
#include <mppi/controllers/R-MPPI/adaptive_robust_mppi_controller.cuh>
#include <mppi/cost_functions/double_integrator/double_integrator_circle_cost.cuh>
#include <mppi/dynamics/double_integrator/di_dynamics.cuh>
#include <mppi/feedback_controllers/DDP/ddp.cuh>
#include <mppi/feedback_controllers/adaptive/adaptive_ddp.cuh>
#include <mppi/sampling_distributions/gaussian/gaussian.cuh>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace
{
const int T = 50;
const int N = 1024;
using DYN = DoubleIntegratorDynamics;
using COST = DoubleIntegratorCircleCost;
using SAMPLING = mppi::sampling_distributions::GaussianDistribution<DYN::DYN_PARAMS_T>;
using STOCK_FB = DDPFeedback<DYN, T>;
using ADAPTIVE_FB = AdaptiveDDPFeedback<DYN, T, 64>;
using STOCK = RobustMPPIController<DYN, COST, STOCK_FB, T, N, SAMPLING>;
using ADAPTIVE = AdaptiveRobustMPPIController<DYN, COST, ADAPTIVE_FB, T, N, SAMPLING>;

const float DT = 0.02f;
const float LAMBDA = 4.0f;
const float ALPHA = 0.0f;
const float VALUE_FUNCTION_THRESHOLD = 10.0f;
const float FAILURE_STATE_COST = 1000.0f;  // same failure criterion as tests/controllers/rmppi_test.cu
const int DISTURBANCE_STEP = 50;

struct RunResult
{
  bool success = true;
  float peak_tracking_error = 0.0f;  // max ||p_actual - p_nominal|| [m]
  float cumulative_cost = 0.0f;      // sum of true-plant state cost per cycle
  float latency_p95_ms = 0.0f;
  int refreshes = 0;
  int overflows = 0;
  int violations = 0;
};

template <class FB>
void setFbParams(FB& fb)
{
  auto p = fb.getParams();
  p.Q.diagonal() << 500, 500, 100, 100;
  p.Q_f = DYN::dfdx::Identity();
  p.R.setIdentity();
  fb.setParams(p);
}

template <class CONTROLLER>
void configure(CONTROLLER& c, unsigned seed)
{
  auto p = c.getParams();
  p.dynamics_rollout_dim_ = dim3(64, 4, 2);
  p.cost_rollout_dim_ = dim3(T, 1, 2);
  p.eval_dyn_kernel_dim_ = dim3(64, 4, 1);
  p.eval_cost_kernel_dim_ = dim3(T, 1, 1);
  p.seed_ = seed;
  c.setParams(p);
  c.setKernelChoice(kernelType::USE_SINGLE_KERNEL);
}

template <class CONTROLLER, class STATS_FN>
RunResult run(CONTROLLER& c, DYN& model, COST& cost, int steps, float severity, STATS_FN stats)
{
  RunResult r;
  std::vector<float> latency;
  DYN::state_array x, next, der;
  DYN::output_array y;
  x << 2, 0, 0, 1;
  for (int k = 0; k < steps; k++)
  {
    if (k == DISTURBANCE_STEP)
    {
      x(2) += severity;
    }
    auto t0 = std::chrono::steady_clock::now();
    c.updateImportanceSamplingControl(x, 1);
    c.computeControl(x, 1);
    latency.push_back(std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count());
    stats(r);
    // RMPPI applies u*_0 + K_0 (x - x*_0): its control sequence is optimized for the nominal state x*, which can
    // differ from the plant state, so the tracking feedback must be applied on the real system too.
    DYN::control_array u = c.getControlSeq().col(0) + c.getFeedbackControl(x, c.getTargetStateSeq().col(0), 0);
    model.step(x, next, der, u, y, k, DT);
    x = next;
    float state_cost = cost.computeStateCost(x);
    r.cumulative_cost += state_cost;
    r.success = r.success && state_cost < FAILURE_STATE_COST && x.allFinite();
    r.peak_tracking_error = std::max(r.peak_tracking_error, (x.head<2>() - c.getNominalState().template head<2>()).norm());
  }
  std::sort(latency.begin(), latency.end());
  r.latency_p95_ms = latency[static_cast<size_t>(0.95 * (latency.size() - 1))];
  return r;
}

void writeRow(std::ofstream& out, float severity, const std::string& mode, int seed, const RunResult& r)
{
  out << severity << "," << mode << "," << seed << "," << r.success << "," << r.peak_tracking_error << ","
      << r.cumulative_cost << "," << r.latency_p95_ms << "," << r.violations << "," << r.refreshes << ","
      << r.overflows << "\n";
}
}  // namespace

int main(int argc, char** argv)
{
  const int num_seeds = argc > 1 ? std::stoi(argv[1]) : 30;
  const int steps = argc > 2 ? std::stoi(argv[2]) : 300;
  const std::string out_path = argc > 3 ? argv[3] : "adaptive_rmppi_benchmark.csv";
  const float radius = argc > 4 ? std::stof(argv[4]) : 0.5f;
  const std::vector<float> severities = { 0.5f, 1.0f, 2.0f };  // velocity kick [m/s]
  // Evaluation seeds start at 1000; tune thresholds on seeds < 1000 only.
  const unsigned seed_offset = 1000;

  std::ofstream out(out_path);
  out << "severity,mode,seed,success,peak_tracking_error,cumulative_cost,latency_p95_ms,violations,refreshes,"
         "overflows\n";
  for (float severity : severities)
  {
    for (int s = 0; s < num_seeds; s++)
    {
      const unsigned seed = seed_offset + s;
      {
        DYN model(100);
        COST cost;
        SAMPLING::SAMPLING_PARAMS_T sp;
        sp.std_dev[0] = sp.std_dev[1] = 1.0f;
        SAMPLING sampler(sp);
        STOCK_FB fb(&model, DT);
        setFbParams(fb);
        STOCK c(&model, &cost, &fb, &sampler, DT, 1, LAMBDA, ALPHA, VALUE_FUNCTION_THRESHOLD);
        configure(c, seed);
        writeRow(out, severity, "stock", seed, run(c, model, cost, steps, severity, [](RunResult&) {}));
      }
      {
        DYN model(100);
        COST cost;
        SAMPLING::SAMPLING_PARAMS_T sp;
        sp.std_dev[0] = sp.std_dev[1] = 1.0f;
        SAMPLING sampler(sp);
        ADAPTIVE_FB fb(&model, DT);
        setFbParams(fb);
        ADAPTIVE c(&model, &cost, &fb, &sampler, DT, 1, LAMBDA, ALPHA, VALUE_FUNCTION_THRESHOLD);
        configure(c, seed);
        ADAPTIVE_FB::Config cfg;
        cfg.enabled = true;
        cfg.distance_radius = radius;
        c.setAdaptiveConfig(cfg);
        auto stats = [&c](RunResult& r) {
          const auto& st = c.getLastIterationStats();
          r.refreshes += st.refreshes;
          r.overflows += st.budget_overflows;
          r.violations += st.violations;
        };
        writeRow(out, severity, "adaptive_v0", seed, run(c, model, cost, steps, severity, stats));
      }
      std::cout << "severity " << severity << " seed " << seed << " done" << std::endl;
    }
  }
  std::cout << "wrote " << out_path << std::endl;
  return 0;
}
