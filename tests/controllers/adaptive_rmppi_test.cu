/**
 * GPU tests for AdaptiveRobustMPPIController. These require a CUDA device.
 */
#include <gtest/gtest.h>
#include <mppi/controllers/R-MPPI/adaptive_robust_mppi_controller.cuh>
#include <mppi/cost_functions/double_integrator/double_integrator_circle_cost.cuh>
#include <mppi/dynamics/double_integrator/di_dynamics.cuh>
#include <mppi/feedback_controllers/DDP/ddp.cuh>
#include <mppi/feedback_controllers/adaptive/adaptive_ddp.cuh>
#include <mppi/sampling_distributions/gaussian/gaussian.cuh>

namespace
{
const int T = 50;
const int N = 256;
using DYN = DoubleIntegratorDynamics;
using COST = DoubleIntegratorCircleCost;
using SAMPLING = mppi::sampling_distributions::GaussianDistribution<DYN::DYN_PARAMS_T>;
using STOCK_FB = DDPFeedback<DYN, T>;
using ADAPTIVE_FB = AdaptiveDDPFeedback<DYN, T, 32>;
using STOCK = RobustMPPIController<DYN, COST, STOCK_FB, T, N, SAMPLING>;
using ADAPTIVE_BASE = AdaptiveRobustMPPIController<DYN, COST, ADAPTIVE_FB, T, N, SAMPLING>;

class AdaptiveUnderTest : public ADAPTIVE_BASE
{
public:
  using ADAPTIVE_BASE::ADAPTIVE_BASE;
  control_trajectory nominalControl() const
  {
    return this->nominal_control_trajectory_;
  }
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

struct Rig
{
  DYN model{ 100 };
  COST cost;
  float dt = 0.02f;
  SAMPLING sampler;
  explicit Rig(float std_dev)
  {
    SAMPLING::SAMPLING_PARAMS_T sp;
    sp.std_dev[0] = std_dev;
    sp.std_dev[1] = std_dev;
    sampler = SAMPLING(sp);
  }
};

// Closed-loop run returning the first control of every cycle.
template <class CONTROLLER>
std::vector<DYN::control_array> runClosedLoop(CONTROLLER& c, DYN& model, float dt, int steps)
{
  std::vector<DYN::control_array> out;
  DYN::state_array x, next, der;
  DYN::output_array y;
  x << 2, 0, 0, 1;
  for (int k = 0; k < steps; k++)
  {
    c.updateImportanceSamplingControl(x, 1);
    c.computeControl(x, 1);
    DYN::control_array u = c.getControlSeq().col(0);
    out.push_back(u);
    model.step(x, next, der, u, y, k, dt);
    x = next;
  }
  return out;
}
}  // namespace

TEST(AdaptiveRMPPI, DisabledMatchesStockRMPPIWithSameSeed)
{
  Rig a(1.0f), b(1.0f);
  STOCK_FB stock_fb(&a.model, a.dt);
  ADAPTIVE_FB adaptive_fb(&b.model, b.dt);
  setFbParams(stock_fb);
  setFbParams(adaptive_fb);
  STOCK stock(&a.model, &a.cost, &stock_fb, &a.sampler, a.dt, 1, 4.0f, 0.0f, 10.0f);
  AdaptiveUnderTest adaptive(&b.model, &b.cost, &adaptive_fb, &b.sampler, b.dt, 1, 4.0f, 0.0f, 10.0f);
  configure(stock, 42);
  configure(adaptive, 42);
  auto u_stock = runClosedLoop(stock, a.model, a.dt, 10);
  auto u_adaptive = runClosedLoop(adaptive, b.model, b.dt, 10);
  for (size_t k = 0; k < u_stock.size(); k++)
  {
    EXPECT_TRUE(u_adaptive[k].isApprox(u_stock[k], 1e-4f)) << "cycle " << k;
  }
  EXPECT_EQ(adaptive.getAccumulatedStats().passes, 0);
}

TEST(AdaptiveRMPPI, EnabledWithUnreachableRadiusMatchesStock)
{
  Rig a(1.0f), b(1.0f);
  STOCK_FB stock_fb(&a.model, a.dt);
  ADAPTIVE_FB adaptive_fb(&b.model, b.dt);
  setFbParams(stock_fb);
  setFbParams(adaptive_fb);
  STOCK stock(&a.model, &a.cost, &stock_fb, &a.sampler, a.dt, 1, 4.0f, 0.0f, 10.0f);
  AdaptiveUnderTest adaptive(&b.model, &b.cost, &adaptive_fb, &b.sampler, b.dt, 1, 4.0f, 0.0f, 10.0f);
  configure(stock, 7);
  configure(adaptive, 7);
  ADAPTIVE_FB::Config cfg;
  cfg.enabled = true;
  cfg.distance_radius = 1e9f;
  adaptive.setAdaptiveConfig(cfg);
  auto u_stock = runClosedLoop(stock, a.model, a.dt, 10);
  auto u_adaptive = runClosedLoop(adaptive, b.model, b.dt, 10);
  for (size_t k = 0; k < u_stock.size(); k++)
  {
    EXPECT_TRUE(u_adaptive[k].isApprox(u_stock[k], 1e-4f)) << "cycle " << k;
  }
  EXPECT_EQ(adaptive.getAccumulatedStats().violations, 0);
  EXPECT_EQ(adaptive.getAccumulatedStats().passes, 10);
}

TEST(AdaptiveRMPPI, RefreshesAreBoundedAndLogged)
{
  Rig b(1.0f);
  ADAPTIVE_FB adaptive_fb(&b.model, b.dt);
  setFbParams(adaptive_fb);
  AdaptiveUnderTest adaptive(&b.model, &b.cost, &adaptive_fb, &b.sampler, b.dt, 1, 4.0f, 0.0f, 10.0f);
  configure(adaptive, 3);
  ADAPTIVE_FB::Config cfg;
  cfg.enabled = true;
  cfg.distance_radius = 0.05f;
  cfg.max_refreshes_per_rollout = 2;
  cfg.max_refreshes_per_iteration = 8;
  adaptive.setAdaptiveConfig(cfg);
  auto u = runClosedLoop(adaptive, b.model, b.dt, 3);
  const auto& s = adaptive.getLastIterationStats();
  EXPECT_GT(s.violations, 0);
  EXPECT_LE(s.refreshes + s.solver_failures + s.horizon_too_short, cfg.max_refreshes_per_iteration);
  EXPECT_LE(s.passes, cfg.max_refreshes_per_rollout + 1);
  EXPECT_GT(s.budget_overflows, 0);
  EXPECT_EQ(static_cast<int>(s.events.size()),
            s.refreshes + s.solver_failures + s.horizon_too_short + s.budget_overflows);
  for (const auto& uk : u)
  {
    EXPECT_TRUE(uk.allFinite());
  }
}

// With (near) zero sampling noise every rollout is the same deterministic pair, so the GPU's first violation and
// refresh must match the CPU reference exactly in (sample, timestep).
TEST(AdaptiveRMPPI, FirstViolationMatchesCPUReference)
{
  Rig b(1e-6f);
  ADAPTIVE_FB adaptive_fb(&b.model, b.dt);
  setFbParams(adaptive_fb);
  AdaptiveUnderTest adaptive(&b.model, &b.cost, &adaptive_fb, &b.sampler, b.dt, 1, 4.0f, 0.0f, 10.0f);
  configure(adaptive, 11);
  ADAPTIVE_FB::Config cfg;
  cfg.enabled = true;
  cfg.distance_radius = 0.3f;
  cfg.max_refreshes_per_rollout = 1;
  cfg.max_refreshes_per_iteration = N;
  adaptive.setAdaptiveConfig(cfg);

  DYN::state_array x;
  x << 2, 0, 0, 1;
  adaptive.updateImportanceSamplingControl(x, 1);
  DYN::state_array x_nom0 = adaptive.getNominalState();
  DYN::state_array x_act0 = x;
  x_act0(0) += 0.5f;  // disturbance the nominal does not see
  auto U = adaptive.nominalControl();

  auto& ref = adaptive_fb.getReference();
  ref.resetIterationBudget();
  auto oracle = ref.rollout(0, x_nom0, x_act0, U, U);
  ASSERT_FALSE(oracle.events.empty());

  adaptive.computeControl(x_act0, 1);
  const auto& s = adaptive.getLastIterationStats();
  ASSERT_FALSE(s.events.empty());
  // Every sample is identical up to 1e-6 noise; at most the budget limits which ones refresh.
  int matching = 0;
  for (const auto& e : s.events)
  {
    if (e.outcome == mppi::adaptive::RefreshOutcome::REFRESHED)
    {
      EXPECT_EQ(e.t, oracle.events[0].t);
      EXPECT_NEAR(e.distance, oracle.events[0].distance, 1e-3f);
      matching++;
    }
  }
  EXPECT_GT(matching, 0);
}
