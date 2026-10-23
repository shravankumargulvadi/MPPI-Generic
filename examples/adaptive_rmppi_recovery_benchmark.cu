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
 *        [seed_offset=1000] [kicks=1,2,3]
 * Writes one summary row per (scenario, severity, mode, seed) and, unless trace is "-", one trace row per cycle.
 * Calibrate kicks and radius with seed_offset < 1000 (e.g. 0) and then freeze them before the evaluation run.
 * Optional experiment settings after the kick list: --force=10 --cart-weight=50 --noise=5 --lambda=0.25
 * --ddp-iterations=1 --budget=64 --modes=both|stock|adaptive --scenarios=all|nominal,kick,mismatch
 * --diagnostics records relative policy-gain changes outside the planning timer. Horizon is set at compile time
 * with ADAPTIVE_CARTPOLE_HORIZON (default 50). The primary success criterion is fixed across experiments.
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
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
#ifndef ADAPTIVE_CARTPOLE_HORIZON
#define ADAPTIVE_CARTPOLE_HORIZON 50
#endif
const int T = ADAPTIVE_CARTPOLE_HORIZON;
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

// Experiment controls. Defaults reproduce the original benchmark settings.
struct Experiment
{
  float force = MAX_FORCE;
  float cart_weight = 50.0f;
  float noise = SAMPLING_STD_DEV;
  float lambda = LAMBDA;
  int ddp_iterations = 1;
  int budget = 64;
  bool diagnostics = false;
  std::string modes = "both";
  std::string scenarios = "all";
} experiment;

bool includeScenario(const std::string& name)
{
  return experiment.scenarios == "all" || ("," + experiment.scenarios + ",").find("," + name + ",") != std::string::npos;
}

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
  float refreshed_weight_mass = 0.0f;
  float gain_change_sum = 0.0f;
  float gain_change_max = 0.0f;
};

struct RunResult
{
  bool success = false;
  bool failed_constraint = false;
  bool pole_success = false;
  float pole_recovery_time_s = -1.0f;
  float peak_cart_position = 0.0f;
  float peak_pole_angle = 0.0f;
  float saturation_fraction = 0.0f;
  float nominal_gap_mean = 0.0f;
  float feedback_abs_mean = 0.0f;
  float refreshed_weight_mass_mean = 0.0f;
  float refreshed_weight_mass_max = 0.0f;
  float refreshed_weight_mass_active_mean = 0.0f;
  float gain_change_mean = 0.0f;
  float gain_change_max = 0.0f;
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
  model.control_rngs_->x = -experiment.force;
  model.control_rngs_->y = experiment.force;
}

void setupCost(COST& cost)
{
  CartpoleQuadraticCostParams p;
  p.cart_position_coeff = experiment.cart_weight;
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
  p.Q.diagonal() << experiment.cart_weight, 10, 200, 1;
  p.Q_f = p.Q;
  p.R.setIdentity();
  p.num_iterations = experiment.ddp_iterations;
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
  int refresh_cycles = 0;
  int last_pole_violation_k = -1;
  bool pole_failed = false;
  int saturation_cycles = 0;
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
    const float pole_angle = std::fabs(wrapAngle(x(2) - static_cast<float>(M_PI)));
    const float pole_error = std::sqrt(pole_angle * pole_angle + 0.1f * x(3) * x(3));
    if (k >= DISTURBANCE_STEP)
    {
      r.peak_error = std::max(r.peak_error, e);
      r.peak_cart_position = std::max(r.peak_cart_position, std::fabs(x(0)));
      r.peak_pole_angle = std::max(r.peak_pole_angle, pole_angle);
      if (!(pole_error < RECOVERY_TOL)) last_pole_violation_k = k;
      if (!(e < RECOVERY_TOL))
      {
        last_violation_k = k;
      }
    }
    if (!x.allFinite() || std::fabs(wrapAngle(x(2) - static_cast<float>(M_PI))) >= M_PI / 2 ||
        std::fabs(x(0)) > TRACK_LIMIT)
    {
      r.failed_constraint = true;
      pole_failed = !x.allFinite() || pole_angle >= M_PI / 2;
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
    refresh_cycles += cs.refreshes > 0;
    r.refreshed_weight_mass_mean += cs.refreshed_weight_mass;
    r.refreshed_weight_mass_max = std::max(r.refreshed_weight_mass_max, cs.refreshed_weight_mass);
    r.gain_change_mean += cs.gain_change_sum;
    r.gain_change_max = std::max(r.gain_change_max, cs.gain_change_max);
    const auto target = c.getTargetStateSeq().col(0).eval();
    float gap_sq = 0.0f;
    for (int i = 0; i < DYN::STATE_DIM; ++i)
    {
      const float d = i == 2 ? wrapAngle(x(i) - target(i)) : x(i) - target(i);
      gap_sq += RECOVERY_WEIGHTS[i] * d * d;
    }
    const float nominal_gap = std::sqrt(gap_sq);
    const float feedforward = c.getControlSeq()(0, 0);
    const float feedback = c.getFeedbackControl(x, target, 0)(0);
    const float requested = feedforward + feedback;
    const bool saturated = std::fabs(requested) > experiment.force;
    saturation_cycles += saturated;
    r.nominal_gap_mean += nominal_gap;
    r.feedback_abs_mean += std::fabs(feedback);
    if (trace)
    {
      *trace << sc.name << "," << sc.severity << "," << mode << "," << seed << "," << k << "," << k * DT << ","
             << x(0) << "," << x(1) << "," << x(2) << "," << x(3) << "," << e << "," << ms << "," << cs.refreshes
             << "," << cs.overflows << "," << cs.violations << "," << feedforward << "," << feedback
             << "," << std::max(-experiment.force, std::min(experiment.force, requested)) << "," << saturated
             << "," << nominal_gap << "," << pole_error << "," << cs.gain_change_sum << "," << cs.gain_change_max << "," << cs.refreshed_weight_mass << "\n";
    }
    // RMPPI applies u*_0 + K_0 (x - x*_0): its control sequence is optimized for the nominal state x*, which can
    // differ from the plant state, so the tracking feedback must be applied on the real system too.
    DYN::control_array u;
    u << requested;
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
  if (!pole_failed)
  {
    r.pole_recovery_time_s = (last_pole_violation_k < 0 ? 0 : last_pole_violation_k + 1 - DISTURBANCE_STEP) * DT;
    r.pole_success = last_pole_violation_k < steps - 1 && r.pole_recovery_time_s <= RECOVERY_DEADLINE_S;
  }
  if (refresh_cycles) r.refreshed_weight_mass_active_mean = r.refreshed_weight_mass_mean / refresh_cycles;
  if (r.refreshes) r.gain_change_mean /= r.refreshes;
  if (!latency.empty())
  {
    r.saturation_fraction = static_cast<float>(saturation_cycles) / latency.size();
    r.refreshed_weight_mass_mean /= latency.size();
    r.nominal_gap_mean /= latency.size();
    r.feedback_abs_mean /= latency.size();
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
      << r.overflows << "," << r.pole_success << "," << r.pole_recovery_time_s << "," << r.peak_cart_position
      << "," << r.peak_pole_angle << "," << r.saturation_fraction << "," << r.nominal_gap_mean << ","
      << r.feedback_abs_mean << "," << r.gain_change_mean << "," << r.gain_change_max << "," << experiment.force
      << "," << experiment.cart_weight << "," << experiment.noise << "," << experiment.lambda << ","
      << experiment.ddp_iterations << "," << experiment.budget << "," << T << "," << r.refreshed_weight_mass_mean
      << "," << r.refreshed_weight_mass_max << "," << r.refreshed_weight_mass_active_mean << "\n";
}

RunResult runStock(const Scenario& sc, unsigned seed, int steps, std::ofstream* trace)
{
  DYN model;
  setupModel(model);
  COST cost;
  setupCost(cost);
  SAMPLING::SAMPLING_PARAMS_T sp;
  sp.std_dev[0] = experiment.noise;
  SAMPLING sampler(sp);
  STOCK_FB fb(&model, DT);
  setFbParams(fb);
  STOCK c(&model, &cost, &fb, &sampler, DT, 1, experiment.lambda, ALPHA, VALUE_FUNCTION_THRESHOLD);
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
  sp.std_dev[0] = experiment.noise;
  SAMPLING sampler(sp);
  ADAPTIVE_FB fb(&model, DT);
  setFbParams(fb);
  ADAPTIVE c(&model, &cost, &fb, &sampler, DT, 1, experiment.lambda, ALPHA, VALUE_FUNCTION_THRESHOLD);
  configure(c, seed);
  ADAPTIVE_FB::Config cfg;
  cfg.enabled = true;
  cfg.distance_radius = radius;
  cfg.max_refreshes_per_iteration = experiment.budget;
  for (int i = 0; i < DYN::STATE_DIM; i++)
  {
    cfg.state_weights[i] = RECOVERY_WEIGHTS[i];
  }
  cfg.angle_state_mask[2] = 1;
  c.setAdaptiveConfig(cfg);
  auto stats = [&c, &fb](CycleStats& s) {
    const auto& st = c.getLastIterationStats();
    s.refreshes = st.refreshes;
    s.overflows = st.budget_overflows;
    s.violations = st.violations;
    if (experiment.diagnostics)
    {
      std::vector<int> previous_slot(N, -1);
      std::vector<bool> refreshed(N, false);
      int slot = 0;
      for (const auto& e : st.events)
      {
        if (e.outcome != mppi::adaptive::RefreshOutcome::REFRESHED) continue;
        refreshed[e.sample] = true;
        double change_sq = 0.0, base_sq = 0.0;
        for (int t = e.t; t < T; ++t)
        {
          const auto old_gain = fb.getPolicyGain(previous_slot[e.sample], t);
          const auto new_gain = fb.getPolicyGain(slot, t);
          change_sq += (new_gain - old_gain).squaredNorm();
          base_sq += old_gain.squaredNorm();
        }
        const float relative_change = std::sqrt(change_sq / std::max(base_sq, 1e-12));
        s.gain_change_sum += relative_change;
        s.gain_change_max = std::max(s.gain_change_max, relative_change);
        previous_slot[e.sample] = slot++;
      }
      // computeControl leaves normalized-exponential actual-system weights in this host array.
      const auto weights = c.getSampledCostSeq();
      double total_weight = 0.0, refreshed_weight = 0.0;
      for (int n = 0; n < N; ++n)
      {
        total_weight += weights(n);
        if (refreshed[n]) refreshed_weight += weights(n);
      }
      s.refreshed_weight_mass = total_weight > 0 ? refreshed_weight / total_weight : 0.0;
    }
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
  // Evaluation seeds start at 1000; tune thresholds and kicks on seeds < 1000 only.
  const unsigned seed_offset = argc > 6 ? static_cast<unsigned>(std::stoul(argv[6])) : 1000;
  std::vector<float> kicks = { 1.0f, 2.0f, 3.0f };
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
    std::string arg = argv[i];
    const size_t eq = arg.find('=');
    const std::string key = arg.substr(0, eq), value = eq == std::string::npos ? "" : arg.substr(eq + 1);
    if (key == "--force") experiment.force = std::stof(value);
    else if (key == "--cart-weight") experiment.cart_weight = std::stof(value);
    else if (key == "--noise") experiment.noise = std::stof(value);
    else if (key == "--lambda") experiment.lambda = std::stof(value);
    else if (key == "--ddp-iterations") experiment.ddp_iterations = std::stoi(value);
    else if (key == "--budget") experiment.budget = std::stoi(value);
    else if (key == "--diagnostics") experiment.diagnostics = true;
    else if (key == "--modes") experiment.modes = value;
    else if (key == "--scenarios") experiment.scenarios = value;
    else throw std::invalid_argument("Unknown experiment option: " + arg);
  }
  if (num_seeds < 1 || steps <= DISTURBANCE_STEP || !(radius > 0) || !(experiment.force > 0) ||
      !(experiment.cart_weight > 0) || !(experiment.noise > 0) || !(experiment.lambda > 0) ||
      experiment.ddp_iterations < 1 || experiment.budget < 0 || experiment.budget > 64 ||
      (experiment.modes != "both" && experiment.modes != "stock" && experiment.modes != "adaptive"))
    throw std::invalid_argument("Invalid experiment settings");

  std::vector<Scenario> scenarios = { { "nominal", 0.0f, false } };
  for (float s : kicks)
  {
    scenarios.push_back({ "kick", s, false });
  }
  for (float s : kicks)
  {
    scenarios.push_back({ "mismatch", s, true });
  }

  std::ofstream out(out_path);
  if (!out) throw std::runtime_error("Cannot write summary: " + out_path);
  out << std::setprecision(10);
  out << "scenario,severity,mode,seed,order,success,failed_constraint,recovery_time_s,peak_error,cumulative_cost,"
         "latency_p95_ms,latency_max_ms,deadline_misses,refreshes_mean,refreshes_p95,budget_hit_fraction,violations,"
         "refreshes,overflows,pole_success,pole_recovery_time_s,peak_cart_position,peak_pole_angle,saturation_fraction,"
         "nominal_gap_mean,feedback_abs_mean,gain_change_mean,gain_change_max,force,cart_weight,noise,lambda,"
         "ddp_iterations,refresh_budget,horizon,refreshed_weight_mass_mean,refreshed_weight_mass_max,refreshed_weight_mass_active_mean\n";
  std::ofstream trace_file;
  std::ofstream* trace = nullptr;
  if (trace_path != "-")
  {
    trace_file.open(trace_path);
    if (!trace_file) throw std::runtime_error("Cannot write trace: " + trace_path);
    trace_file << std::setprecision(10);
    trace_file << "scenario,severity,mode,seed,cycle,time_s,x,x_dot,theta,theta_dot,error,latency_ms,refreshes,"
                  "overflows,violations,feedforward,feedback,applied_control,saturated,nominal_gap,pole_error,gain_change_sum,"
                  "gain_change_max,refreshed_weight_mass\n";
    trace = &trace_file;
  }
  for (const auto& sc : scenarios)
  {
    if (!includeScenario(sc.name)) continue;
    for (int s = 0; s < num_seeds; s++)
    {
      const unsigned seed = seed_offset + s;
      const bool stock_first = s % 2 == 0;
      for (int order = 0; order < 2; order++)
      {
        if ((order == 0) == stock_first)
        {
          if (experiment.modes == "adaptive") continue;
          writeRow(out, sc, "stock", seed, order, runStock(sc, seed, steps, trace));
        }
        else
        {
          if (experiment.modes == "stock") continue;
          writeRow(out, sc, "adaptive_v0", seed, order, runAdaptive(sc, seed, steps, radius, trace));
        }
      }
      std::cout << sc.name << " " << sc.severity << " seed " << seed << " done" << std::endl;
    }
  }
  std::cout << "wrote " << out_path << (trace ? " and " + trace_path : "") << std::endl;
  return 0;
}
