/**
 * Paired closed-loop heading-recovery benchmark: stock RMPPI vs adaptive RMPPI (V0) on a Dubins vehicle.
 *
 * Task: follow the line y = 0 heading +x (QuadraticCost on y and yaw only; x is free). At cycle DISTURBANCE_STEP the
 * plant yaw is kicked by `severity` rad. Lateral error obeys y_dot = v sin(yaw), so gains linearized about the nominal
 * heading (yaw ~ 0) have the wrong sign once |yaw| > pi/2, while a policy re-solved at the actual state does not.
 * Both modes see the same seed, plant, cost, horizon, N, lambda, limits and cycle period; order alternates by seed.
 *
 * Recovery metric: e = sqrt(y^2 + wrap(yaw)^2) using the plant state at the start of each cycle. Success: finite
 * state, |y| never exceeds LATERAL_LIMIT, and e < RECOVERY_TOL from some time <= RECOVERY_DEADLINE_S after the kick
 * until the end of the trial. Constants below were fixed before any run; nothing was tuned on evaluation seeds.
 *
 * Usage: adaptive_rmppi_heading_benchmark [num_seeds=30] [steps=300] [summary.csv] [radius=0.5] [trace.csv|-]
 *        [seed_offset=1000] [kicks=1.0,2.0,2.5,3.0]
 * Optional ablation flags after kicks: --plant-feedback=0|1 --rollout-budget=0..64 (defaults 1 and 64).
 * --stock-ddp-action=1 adds the attribution baseline: outside the radius, stock executes the first action from
 * its already computed DDP result without another solve. This is off by default.
 * Requires a CUDA GPU.
 */
#include <mppi/controllers/R-MPPI/adaptive_robust_mppi_controller.cuh>
#include <mppi/cost_functions/quadratic_cost/quadratic_cost.cuh>
#include <mppi/dynamics/dubins/dubins.cuh>
#include <mppi/feedback_controllers/DDP/ddp.cuh>
#include <mppi/feedback_controllers/adaptive/adaptive_ddp.cuh>
#include <mppi/sampling_distributions/gaussian/gaussian.cuh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
bool benchmark_plant_feedback = true;
int benchmark_rollout_budget = 64;
bool benchmark_stock_ddp_action = false;
float benchmark_radius = 0.5f;

const int T = 50;
const int N = 1024;
using DYN = DubinsDynamics;
using COST = QuadraticCost<DYN>;
using SAMPLING = mppi::sampling_distributions::GaussianDistribution<DYN::DYN_PARAMS_T>;
using STOCK_FB = DDPFeedback<DYN, T>;
using ADAPTIVE_FB = AdaptiveDDPFeedback<DYN, T, 64>;
using STOCK = RobustMPPIController<DYN, COST, STOCK_FB, T, N, SAMPLING>;
using ADAPTIVE = AdaptiveRobustMPPIController<DYN, COST, ADAPTIVE_FB, T, N, SAMPLING>;

const float DT = 0.02f;
const float CYCLE_DEADLINE_MS = 1000.0f * DT;
const float LAMBDA = 1.0f;
const float ALPHA = 0.0f;
const float VALUE_FUNCTION_THRESHOLD = 10.0f;
const float STD_DEV_VEL = 0.5f;
const float STD_DEV_YAW_RATE = 1.0f;
const float VEL_MIN = 0.5f, VEL_MAX = 2.0f;
const float YAW_RATE_MAX = 2.0f;
const int DISTURBANCE_STEP = 50;
const float RECOVERY_WEIGHTS[3] = { 0.0f, 1.0f, 1.0f };  // x is free along the line
const int HEADING_ANGLE_MASK[3] = {0, 0, 1};
const float RECOVERY_TOL = 0.2f;
const float RECOVERY_DEADLINE_S = 4.0f;
const float LATERAL_LIMIT = 10.0f;

struct Scenario
{
  std::string name;
  float severity;  // yaw kick [rad]
};

struct CycleStats
{
  int refreshes = 0;
  int plant_refreshes = 0;
  int plant_refresh_failures = 0;
  int overflows = 0;
  int violations = 0;
};

struct RunResult
{
  bool success = false;
  bool failed_constraint = false;
  float recovery_time_s = -1.0f;
  float peak_error = 0.0f;
  float cumulative_cost = 0.0f;
  float latency_p95_ms = 0.0f;
  float latency_max_ms = 0.0f;
  int deadline_misses = 0;
  float refreshes_mean = 0.0f;
  float refreshes_p95 = 0.0f;
  float budget_hit_fraction = 0.0f;
  int refreshes = 0;
  int overflows = 0;
  int violations = 0;
  int plant_refreshes = 0;
  int plant_refresh_failures = 0;
  int reused_ddp_actions = 0;
};

float wrapAngle(float a)
{
  return std::atan2(std::sin(a), std::cos(a));
}

float recoveryError(const DYN::state_array& x)
{
  const float d[3] = { x(0), x(1), wrapAngle(x(2)) };
  float s = 0.0f;
  for (int i = 0; i < 3; i++)
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
  model.control_rngs_[0].x = VEL_MIN;
  model.control_rngs_[0].y = VEL_MAX;
  model.control_rngs_[1].x = -YAW_RATE_MAX;
  model.control_rngs_[1].y = YAW_RATE_MAX;
}

void setupCost(COST& cost)
{
  auto p = cost.getParams();
  p.s_goal[0] = p.s_goal[1] = p.s_goal[2] = 0.0f;
  p.s_coeffs[0] = 0.0f;   // x free
  p.s_coeffs[1] = 10.0f;  // lateral offset
  p.s_coeffs[2] = 2.0f;   // yaw
  p.control_cost_coeff[0] = 0.0f;
  p.control_cost_coeff[1] = 0.0f;
  cost.setParams(p);
}

template <class FB>
void setFbParams(FB& fb)
{
  auto p = fb.getParams();
  p.Q.setZero();
  p.Q.diagonal() << 0, 10, 2;
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
              const std::string& mode, unsigned seed, STOCK_FB* reuse_ddp = nullptr)
{
  DYN plant;
  setupModel(plant);
  RunResult r;
  std::vector<float> latency, refreshes_per_cycle;
  int budget_hits = 0;
  int last_violation_k = -1;
  DYN::state_array x, next, der;
  DYN::output_array y;
  x << 0, 0, 0;
  for (int k = 0; k < steps; k++)
  {
    if (k == DISTURBANCE_STEP)
    {
      x(2) = wrapAngle(x(2) + sc.severity);
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
    if (!x.allFinite() || std::fabs(x(1)) > LATERAL_LIMIT)
    {
      r.failed_constraint = true;
      break;
    }
    auto t0 = std::chrono::steady_clock::now();
    c.updateImportanceSamplingControl(x, 1);
    c.computeControl(x, 1);
    const auto target = c.getTargetStateSeq().col(0).eval();
    const auto feedforward = c.getControlSeq().col(0).eval();
    auto feedback = c.getFeedbackControl(x, target, 0);
    bool reused_ddp_action = false;
    if (reuse_ddp && benchmark_stock_ddp_action && mppi::adaptive::weightedStateDistance<DYN::STATE_DIM>(
            x.data(), target.data(), RECOVERY_WEIGHTS, HEADING_ANGLE_MASK) > benchmark_radius)
    {
      feedback = reuse_ddp->result_.control_trajectory.col(0) - feedforward;
      reused_ddp_action = true;
      r.reused_ddp_actions++;
    }
    DYN::control_array u = feedforward + feedback;
    const float ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
    latency.push_back(ms);
    r.deadline_misses += ms > CYCLE_DEADLINE_MS;
    CycleStats cs;
    stats(cs);
    r.refreshes += cs.refreshes;
    r.overflows += cs.overflows;
    r.violations += cs.violations;
    r.plant_refreshes += cs.plant_refreshes;
    r.plant_refresh_failures += cs.plant_refresh_failures;
    refreshes_per_cycle.push_back(static_cast<float>(cs.refreshes));
    budget_hits += cs.overflows > 0;
    if (trace)
    {
      *trace << sc.name << "," << sc.severity << "," << mode << "," << seed << "," << k << "," << k * DT << ","
             << x(0) << "," << x(1) << "," << x(2) << "," << u(0) << "," << u(1) << "," << e << "," << ms << ","
             << cs.refreshes << "," << cs.plant_refreshes << "," << cs.overflows << "," << cs.violations << "," << target(1) << "," << target(2)
             << "," << feedforward(0) << "," << feedforward(1) << "," << feedback(0) << "," << feedback(1)
             << "," << std::max(VEL_MIN, std::min(VEL_MAX, u(0)))
             << "," << std::max(-YAW_RATE_MAX, std::min(YAW_RATE_MAX, u(1)))
             << "," << cs.plant_refresh_failures << "," << reused_ddp_action << "\n";
    }
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
      << r.overflows << "," << r.plant_refreshes << "," << r.plant_refresh_failures
      << "," << benchmark_plant_feedback << "," << benchmark_rollout_budget << "," << benchmark_stock_ddp_action
      << "," << r.reused_ddp_actions << "\n";
}

RunResult runStock(const Scenario& sc, unsigned seed, int steps, std::ofstream* trace)
{
  DYN model;
  setupModel(model);
  COST cost;
  setupCost(cost);
  SAMPLING::SAMPLING_PARAMS_T sp;
  sp.std_dev[0] = STD_DEV_VEL;
  sp.std_dev[1] = STD_DEV_YAW_RATE;
  SAMPLING sampler(sp);
  STOCK_FB fb(&model, DT);
  setFbParams(fb);
  STOCK c(&model, &cost, &fb, &sampler, DT, 1, LAMBDA, ALPHA, VALUE_FUNCTION_THRESHOLD);
  configure(c, seed);
  return run(c, sc, cost, steps, [](CycleStats&) {}, trace, "stock", seed, &fb);
}

RunResult runAdaptive(const Scenario& sc, unsigned seed, int steps, float radius, std::ofstream* trace)
{
  DYN model;
  setupModel(model);
  COST cost;
  setupCost(cost);
  SAMPLING::SAMPLING_PARAMS_T sp;
  sp.std_dev[0] = STD_DEV_VEL;
  sp.std_dev[1] = STD_DEV_YAW_RATE;
  SAMPLING sampler(sp);
  ADAPTIVE_FB fb(&model, DT);
  setFbParams(fb);
  ADAPTIVE c(&model, &cost, &fb, &sampler, DT, 1, LAMBDA, ALPHA, VALUE_FUNCTION_THRESHOLD);
  configure(c, seed);
  ADAPTIVE_FB::Config cfg;
  cfg.enabled = true;
  cfg.distance_radius = radius;
  cfg.plant_feedback = benchmark_plant_feedback;
  cfg.max_refreshes_per_iteration = benchmark_rollout_budget;
  for (int i = 0; i < DYN::STATE_DIM; i++)
  {
    cfg.state_weights[i] = RECOVERY_WEIGHTS[i];
  }
  cfg.angle_state_mask[2] = 1;
  c.setAdaptiveConfig(cfg);
  auto stats = [&c](CycleStats& s) {
    const auto& st = c.getLastIterationStats();
    s.refreshes = st.refreshes;
    s.plant_refreshes = st.plant_refreshes;
    s.plant_refresh_failures = st.plant_refresh_failures;
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
  const std::string out_path = argc > 3 ? argv[3] : "adaptive_rmppi_heading_benchmark.csv";
  const float radius = argc > 4 ? std::stof(argv[4]) : 0.5f;
  benchmark_radius = radius;
  const std::string trace_path = argc > 5 ? argv[5] : "adaptive_rmppi_heading_trace.csv";
  const unsigned seed_offset = argc > 6 ? static_cast<unsigned>(std::stoul(argv[6])) : 1000;
  std::vector<float> kicks = { 1.0f, 2.0f, 2.5f, 3.0f };
  if (argc > 7)
  {
    kicks.clear();
    std::string list = argv[7];
    size_t start = 0;
    while (start <= list.size())
    {
      const size_t end = std::min(list.find(',', start), list.size());
      if (end > start)
      {
        kicks.push_back(std::stof(list.substr(start, end - start)));
      }
      start = end + 1;
    }
  }

  for (int i = 8; i < argc; ++i)
  {
    const std::string option = argv[i];
    if (option == "--plant-feedback=0") benchmark_plant_feedback = false;
    else if (option == "--plant-feedback=1") benchmark_plant_feedback = true;
    else if (option == "--stock-ddp-action=1") benchmark_stock_ddp_action = true;
    else if (option == "--stock-ddp-action=0") benchmark_stock_ddp_action = false;
    else if (option.find("--rollout-budget=") == 0) benchmark_rollout_budget = std::stoi(option.substr(17));
    else throw std::invalid_argument("Unknown heading-benchmark option: " + option);
  }
  if (num_seeds < 1 || steps <= DISTURBANCE_STEP || !(radius > 0) || benchmark_rollout_budget < 0 || benchmark_rollout_budget > 64)
    throw std::invalid_argument("Invalid heading-benchmark settings");

  std::vector<Scenario> scenarios = { { "nominal", 0.0f } };
  for (float s : kicks)
  {
    scenarios.push_back({ "heading_kick", s });
  }

  std::ofstream out(out_path);
  if (!out) throw std::runtime_error("Cannot write summary");
  out << std::setprecision(10);
  out << "scenario,severity,mode,seed,order,success,failed_constraint,recovery_time_s,peak_error,cumulative_cost,"
         "latency_p95_ms,latency_max_ms,deadline_misses,refreshes_mean,refreshes_p95,budget_hit_fraction,violations,"
         "refreshes,overflows,plant_refreshes,plant_refresh_failures,plant_feedback,rollout_budget,stock_ddp_action,reused_ddp_actions\n";
  std::ofstream trace_file;
  std::ofstream* trace = nullptr;
  if (trace_path != "-")
  {
    trace_file.open(trace_path);
    if (!trace_file) throw std::runtime_error("Cannot write trace");
    trace_file << std::setprecision(10);
    trace_file << "scenario,severity,mode,seed,cycle,time_s,x,y,yaw,vel,yaw_rate,error,latency_ms,refreshes,"
                  "plant_refreshes,overflows,violations,nominal_y,nominal_yaw,feedforward_vel,feedforward_yaw_rate,"
                  "feedback_vel,feedback_yaw_rate,applied_vel,applied_yaw_rate,plant_refresh_failures,reused_ddp_action\n";
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
