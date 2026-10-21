/**
 * Paired closed-loop disturbance-recovery benchmark on a nonlinear plant: stock RMPPI vs adaptive RMPPI (V0).
 *
 * Planner model: cartpole (cart 1 kg, pole 1 kg, length 1 m) balancing upright at x = [0, 0, pi, 0].
 * Scenarios (fixed before any evaluation run; nothing below was tuned on evaluation seeds):
 *   nominal   - matched plant, no disturbance.
 *   kick      - matched plant, pole angular-velocity kick of `severity` rad/s at cycle DISTURBANCE_STEP.
 *   mismatch  - same kicks, but the simulated plant has pole mass 1.5 kg and length 1.3 m; the planner is unchanged.
 * Both modes see the same seed, initial state, plant, cost, horizon, N, lambda, actuator limits and cycle period.
 * Mode order alternates by seed to avoid a systematic warm-up advantage.
 *
 * Recovery metric: e = sqrt(sum_i w_i (x_i - x_goal_i)^2) with wrapped pole angle and w = RECOVERY_WEIGHTS.
 * A trial succeeds when the state stays finite, the pole never passes horizontal (|theta - pi| < pi/2), the cart
 * stays within +-TRACK_LIMIT, and e < RECOVERY_TOL from some time <= RECOVERY_DEADLINE_S after the disturbance
 * until the end of the trial. Errors use the plant state at the start of each control cycle (time k * DT).
 *
 * Usage: adaptive_rmppi_recovery_benchmark [num_seeds=30] [steps=300] [summary.csv] [radius=0.5] [trace.csv|-]
 * Writes one summary row per (scenario, severity, mode, seed) and, unless trace is "-", one trace row per cycle.
 * Requires a CUDA GPU.
 */
#include <mppi/controllers/R-MPPI/adaptive_robust_mppi_controller.cuh>
#include <mppi/cost_functions/cartpole/cartpole_quadratic_cost.cuh>
#include <mppi/dynamics/cartpole/cartpole_dynamics.cuh>
#include <mppi/feedback_controllers/DDP/ddp.cuh>
#include <mppi/feedback_controllers/adaptive/adaptive_ddp.cuh>
#include <mppi/sampling_distributions/gaussian/gaussian.cuh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace
{
const int T = 50;
const int N = 1024;
using DYN = CartpoleDynamics;
using COST = CartpoleQuadraticCost;
using SAMPLING = mppi::sampling_distributions::GaussianDistribution<DYN::DYN_PARAMS_T>;
using STOCK_FB = DDPFeedback<DYN, T>;
using ADAPTIVE_FB = AdaptiveDDPFeedback<DYN, T, 64>;
using STOCK = RobustMPPIController<DYN, COST, STOCK_FB, T, N, SAMPLING>;
using ADAPTIVE = AdaptiveRobustMPPIController<DYN, COST, ADAPTIVE_FB, T, N, SAMPLING>;

const float DT = 0.02f;
const float CYCLE_DEADLINE_MS = 1000.0f * DT;
const float LAMBDA = 0.25f;
const float ALPHA = 0.0f;
const float VALUE_FUNCTION_THRESHOLD = 10.0f;
const float SAMPLING_STD_DEV = 5.0f;
const float MAX_FORCE = 10.0f;
const int DISTURBANCE_STEP = 50;
const float RECOVERY_WEIGHTS[4] = { 1.0f, 0.1f, 1.0f, 0.1f };
const float RECOVERY_TOL = 0.2f;
const float RECOVERY_DEADLINE_S = 2.0f;
const float TRACK_LIMIT = 5.0f;
const float MISMATCH_POLE_MASS = 1.5f;
const float MISMATCH_POLE_LENGTH = 1.3f;

struct Scenario
{
  std::string name;
  float severity;  // pole angular-velocity kick [rad/s]
  bool mismatched_plant;
};

struct CycleStats
{
  int refreshes = 0;
  int overflows = 0;
  int violations = 0;
};

struct RunResult
{
  bool success = false;
  bool failed_constraint = false;
  float recovery_time_s = -1.0f;
  float peak_error = 0.0f;       // max e over cycles >= DISTURBANCE_STEP
  float cumulative_cost = 0.0f;  // true-plant state cost summed over cycles >= DISTURBANCE_STEP
  float latency_p95_ms = 0.0f;
  float latency_max_ms = 0.0f;
  int deadline_misses = 0;
  float refreshes_mean = 0.0f;
  float refreshes_p95 = 0.0f;
  float budget_hit_fraction = 0.0f;
  int refreshes = 0;
  int overflows = 0;
  int violations = 0;
};

float wrapAngle(float a)
{
  return std::atan2(std::sin(a), std::cos(a));
}

float recoveryError(const DYN::state_array& x)
{
  const float d[4] = { x(0), x(1), wrapAngle(x(2) - static_cast<float>(M_PI)), x(3) };
  float s = 0.0f;
  for (int i = 0; i < 4; i++)
  {
    s += RECOVERY_WEIGHTS[i] * d[i] * d[i];
  }
  return std::sqrt(s);
}

float percentile(std::vector<float> v, float q)
{
  std::sort(v.begin(), v.end());
  return v[static_cast<size_t>(q * (v.size() - 1))];
}

void setupModel(DYN& model)
{
  model.control_rngs_->x = -MAX_FORCE;
  model.control_rngs_->y = MAX_FORCE;
}

void setupCost(COST& cost)
{
  CartpoleQuadraticCostParams p;
  p.cart_position_coeff = 50;
  p.cart_velocity_coeff = 10;
  p.pole_angle_coeff = 200;
  p.pole_angular_velocity_coeff = 1;
  p.control_cost_coeff[0] = 0;
  p.terminal_cost_coeff = 0;
  p.desired_terminal_state[0] = 0;
  p.desired_terminal_state[1] = 0;
  p.desired_terminal_state[2] = M_PI;
  p.desired_terminal_state[3] = 0;
  cost.setParams(p);
}

template <class FB>
void setFbParams(FB& fb)
{
  auto p = fb.getParams();
  p.Q.setZero();
  p.Q.diagonal() << 50, 10, 200, 1;
  p.Q_f = p.Q;
  p.R.setIdentity();
  p.num_iterations = 1;
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
RunResult run(CONTROLLER& c, const Scenario& sc, COST& cost, int steps, STATS_FN stats, std::ofstream* trace,
              const std::string& mode, unsigned seed)
{
  DYN plant(1.0f, sc.mismatched_plant ? MISMATCH_POLE_MASS : 1.0f, sc.mismatched_plant ? MISMATCH_POLE_LENGTH : 1.0f);
  setupModel(plant);
  RunResult r;
  std::vector<float> latency, refreshes_per_cycle;
  int budget_hits = 0;
  int last_violation_k = -1;  // last cycle >= DISTURBANCE_STEP with e >= RECOVERY_TOL
  DYN::state_array x, next, der;
  DYN::output_array y;
  x << 0, 0, M_PI, 0;
  for (int k = 0; k < steps; k++)
  {
    if (k == DISTURBANCE_STEP)
    {
      x(3) += sc.severity;
    }
    const float e = recoveryError(x);
    if (k >= DISTURBANCE_STEP)
    {
      r.peak_error = std::max(r.peak_error, e);
      if (!(e < RECOVERY_TOL))
      {
        last_violation_k = k;
      }
    }
    if (!x.allFinite() || std::fabs(wrapAngle(x(2) - static_cast<float>(M_PI))) >= M_PI / 2 ||
        std::fabs(x(0)) > TRACK_LIMIT)
    {
      r.failed_constraint = true;
      break;
    }
    auto t0 = std::chrono::steady_clock::now();
    c.updateImportanceSamplingControl(x, 1);
    c.computeControl(x, 1);
    const float ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
    latency.push_back(ms);
    r.deadline_misses += ms > CYCLE_DEADLINE_MS;
    CycleStats cs;
    stats(cs);
    r.refreshes += cs.refreshes;
    r.overflows += cs.overflows;
    r.violations += cs.violations;
    refreshes_per_cycle.push_back(static_cast<float>(cs.refreshes));
    budget_hits += cs.overflows > 0;
    if (trace)
    {
      *trace << sc.name << "," << sc.severity << "," << mode << "," << seed << "," << k << "," << k * DT << ","
             << x(0) << "," << x(1) << "," << x(2) << "," << x(3) << "," << e << "," << ms << "," << cs.refreshes
             << "," << cs.overflows << "," << cs.violations << "\n";
    }
    DYN::control_array u = c.getControlSeq().col(0);
    plant.enforceConstraints(x, u);
    plant.step(x, next, der, u, y, k * DT, DT);
    x = next;
    if (k >= DISTURBANCE_STEP)
    {
      r.cumulative_cost += cost.computeStateCost(y);
    }
  }
  if (!r.failed_constraint)
  {
    r.recovery_time_s = (last_violation_k < 0 ? 0 : last_violation_k + 1 - DISTURBANCE_STEP) * DT;
    r.success = last_violation_k < steps - 1 && r.recovery_time_s <= RECOVERY_DEADLINE_S;
  }
  if (!latency.empty())
  {
    r.latency_p95_ms = percentile(latency, 0.95f);
    r.latency_max_ms = *std::max_element(latency.begin(), latency.end());
    r.refreshes_mean = static_cast<float>(r.refreshes) / latency.size();
    r.refreshes_p95 = percentile(refreshes_per_cycle, 0.95f);
    r.budget_hit_fraction = static_cast<float>(budget_hits) / latency.size();
  }
  return r;
}

void writeRow(std::ofstream& out, const Scenario& sc, const std::string& mode, unsigned seed, int order,
              const RunResult& r)
{
  out << sc.name << "," << sc.severity << "," << mode << "," << seed << "," << order << "," << r.success << ","
      << r.failed_constraint << "," << r.recovery_time_s << "," << r.peak_error << "," << r.cumulative_cost << ","
      << r.latency_p95_ms << "," << r.latency_max_ms << "," << r.deadline_misses << "," << r.refreshes_mean << ","
      << r.refreshes_p95 << "," << r.budget_hit_fraction << "," << r.violations << "," << r.refreshes << ","
      << r.overflows << "\n";
}

RunResult runStock(const Scenario& sc, unsigned seed, int steps, std::ofstream* trace)
{
  DYN model;
  setupModel(model);
  COST cost;
  setupCost(cost);
  SAMPLING::SAMPLING_PARAMS_T sp;
  sp.std_dev[0] = SAMPLING_STD_DEV;
  SAMPLING sampler(sp);
  STOCK_FB fb(&model, DT);
  setFbParams(fb);
  STOCK c(&model, &cost, &fb, &sampler, DT, 1, LAMBDA, ALPHA, VALUE_FUNCTION_THRESHOLD);
  configure(c, seed);
  return run(c, sc, cost, steps, [](CycleStats&) {}, trace, "stock", seed);
}

RunResult runAdaptive(const Scenario& sc, unsigned seed, int steps, float radius, std::ofstream* trace)
{
  DYN model;
  setupModel(model);
  COST cost;
  setupCost(cost);
  SAMPLING::SAMPLING_PARAMS_T sp;
  sp.std_dev[0] = SAMPLING_STD_DEV;
  SAMPLING sampler(sp);
  ADAPTIVE_FB fb(&model, DT);
  setFbParams(fb);
  ADAPTIVE c(&model, &cost, &fb, &sampler, DT, 1, LAMBDA, ALPHA, VALUE_FUNCTION_THRESHOLD);
  configure(c, seed);
  ADAPTIVE_FB::Config cfg;
  cfg.enabled = true;
  cfg.distance_radius = radius;
  for (int i = 0; i < DYN::STATE_DIM; i++)
  {
    cfg.state_weights[i] = RECOVERY_WEIGHTS[i];
  }
  cfg.angle_state_mask[2] = 1;
  c.setAdaptiveConfig(cfg);
  auto stats = [&c](CycleStats& s) {
    const auto& st = c.getLastIterationStats();
    s.refreshes = st.refreshes;
    s.overflows = st.budget_overflows;
    s.violations = st.violations;
  };
  return run(c, sc, cost, steps, stats, trace, "adaptive_v0", seed);
}
}  // namespace

int main(int argc, char** argv)
{
  const int num_seeds = argc > 1 ? std::stoi(argv[1]) : 30;
  const int steps = argc > 2 ? std::stoi(argv[2]) : 300;
  const std::string out_path = argc > 3 ? argv[3] : "adaptive_rmppi_recovery_benchmark.csv";
  const float radius = argc > 4 ? std::stof(argv[4]) : 0.5f;
  const std::string trace_path = argc > 5 ? argv[5] : "adaptive_rmppi_recovery_trace.csv";
  // Evaluation seeds start at 1000; tune thresholds on seeds < 1000 only.
  const unsigned seed_offset = 1000;

  std::vector<Scenario> scenarios = { { "nominal", 0.0f, false } };
  for (float s : { 1.0f, 2.0f, 3.0f })
  {
    scenarios.push_back({ "kick", s, false });
  }
  for (float s : { 1.0f, 2.0f, 3.0f })
  {
    scenarios.push_back({ "mismatch", s, true });
  }

  std::ofstream out(out_path);
  out << "scenario,severity,mode,seed,order,success,failed_constraint,recovery_time_s,peak_error,cumulative_cost,"
         "latency_p95_ms,latency_max_ms,deadline_misses,refreshes_mean,refreshes_p95,budget_hit_fraction,violations,"
         "refreshes,overflows\n";
  std::ofstream trace_file;
  std::ofstream* trace = nullptr;
  if (trace_path != "-")
  {
    trace_file.open(trace_path);
    trace_file << "scenario,severity,mode,seed,cycle,time_s,x,x_dot,theta,theta_dot,error,latency_ms,refreshes,"
                  "overflows,violations\n";
    trace = &trace_file;
  }
  for (const auto& sc : scenarios)
  {
    for (int s = 0; s < num_seeds; s++)
    {
      const unsigned seed = seed_offset + s;
      const bool stock_first = s % 2 == 0;
      for (int order = 0; order < 2; order++)
      {
        if ((order == 0) == stock_first)
        {
          writeRow(out, sc, "stock", seed, order, runStock(sc, seed, steps, trace));
        }
        else
        {
          writeRow(out, sc, "adaptive_v0", seed, order, runAdaptive(sc, seed, steps, radius, trace));
        }
      }
      std::cout << sc.name << " " << sc.severity << " seed " << seed << " done" << std::endl;
    }
  }
  std::cout << "wrote " << out_path << (trace ? " and " + trace_path : "") << std::endl;
  return 0;
}
