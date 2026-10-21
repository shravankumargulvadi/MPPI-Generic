#include <gtest/gtest.h>
#include <mppi/dynamics/double_integrator/di_dynamics.cuh>
#include <mppi/dynamics/dubins/dubins.cuh>
#include <mppi/feedback_controllers/adaptive/adaptive_rmppi_reference.h>

#include <random>

using namespace mppi::adaptive;

template <class DYN>
class AdaptiveReferenceFixture
{
public:
  using REF = AdaptiveRMPPIReference<DYN>;
  using state_array = typename DYN::state_array;
  using control_array = typename DYN::control_array;
  using state_traj = typename REF::state_traj;
  using control_traj = typename REF::control_traj;
  using K_T = typename REF::feedback_matrix;
  static const int S = DYN::STATE_DIM;
  static const int C = DYN::CONTROL_DIM;

  DYN model;
  float dt = 0.05f;
  int T = 20;
  DDPParams<DYN> ddp_params;
  AdaptiveFeedbackConfig<S> config;
  control_traj v_nom, v_act;
  state_traj x_ref;
  std::vector<K_T> K;

  explicit AdaptiveReferenceFixture(unsigned seed = 7)
  {
    ddp_params.Q.setIdentity();
    ddp_params.Q_f.setIdentity();
    ddp_params.R.setIdentity();
    ddp_params.num_iterations = 2;
    std::mt19937 gen(seed);
    std::normal_distribution<float> n(0.0f, 0.3f);
    v_nom = control_traj::Constant(C, T, 0.5f);
    v_act = v_nom;
    for (int t = 0; t < T; t++)
    {
      for (int i = 0; i < C; i++)
      {
        v_act(i, t) += n(gen);
      }
    }
    for (int t = 0; t < T; t++)
    {
      K.push_back(-0.5f * K_T::Identity());
    }
  }

  // Open-loop rollout of the nominal controls: the trajectory baseline gains are linearized about.
  state_traj openLoop(const state_array& x0, const control_traj& v)
  {
    state_traj x(S, T + 1);
    x.col(0) = x0;
    state_array s = x0, next, der;
    typename DYN::output_array out;
    for (int t = 0; t < T; t++)
    {
      control_array u = v.col(t);
      model.enforceConstraints(s, u);
      model.step(s, next, der, u, out, t, dt);
      s = next;
      x.col(t + 1) = s;
    }
    return x;
  }

  REF make(const state_array& x0)
  {
    REF ref(&model, dt, T, config, ddp_params);
    x_ref = openLoop(x0, v_nom);
    ref.setBaselineCache(K, x_ref, v_nom);
    ref.setRefreshReference(x_ref, v_nom);
    return ref;
  }

  // Solver stub returning a recognizable constant gain and the actual state as the recovery start.
  typename REF::GainSolver markerSolver(float gain, int* calls = nullptr)
  {
    return [gain, calls](const state_array& x0, int t0, const state_traj& xg, const control_traj& ug,
                         typename REF::Cache& out) {
      if (calls)
      {
        (*calls)++;
      }
      out = typename REF::Cache();
      out.start_t = t0;
      for (int k = 0; k < xg.cols(); k++)
      {
        out.K.push_back(gain * K_T::Identity());
        out.x_ref.push_back(k == 0 ? x0 : state_array(xg.col(k)));
        out.u_ref.push_back(ug.col(k));
      }
      return true;
    };
  }
};

using DubinsFixture = AdaptiveReferenceFixture<DubinsDynamics>;
using DIFixture = AdaptiveReferenceFixture<DoubleIntegratorDynamics>;

TEST(AdaptiveRMPPIReference, DisabledMatchesBaselineTrackingLaw)
{
  DubinsFixture f;
  f.config.enabled = false;
  DubinsFixture::state_array x0 = DubinsFixture::state_array::Zero();
  auto ref = f.make(x0);
  DubinsFixture::state_array x_act0;
  x_act0 << 1.0f, -1.0f, 2.0f;
  auto r = ref.rollout(0, x0, x_act0, f.v_nom, f.v_act);
  EXPECT_TRUE(r.events.empty());
  EXPECT_EQ(r.num_refreshes, 0);
  for (int t = 0; t < f.T; t++)
  {
    DubinsFixture::control_array expected = f.K[t] * (r.x_act.col(t) - r.x_nom.col(t));
    EXPECT_TRUE(r.u_fb.col(t).isApprox(expected, 1e-6f)) << "t = " << t;
  }
}

TEST(AdaptiveRMPPIReference, NominalTrajectoryUnaffectedByRefresh)
{
  DubinsFixture f;
  f.config.enabled = true;
  f.config.distance_radius = 0.01f;
  f.config.max_refreshes_per_rollout = 100;
  f.config.max_refreshes_per_iteration = 100;
  DubinsFixture::state_array x0 = DubinsFixture::state_array::Zero();
  auto ref = f.make(x0);
  ref.setSolver(f.markerSolver(-2.0f));
  DubinsFixture::state_array x_act0;
  x_act0 << 0.5f, 0.5f, 0.5f;
  auto r = ref.rollout(0, x0, x_act0, f.v_nom, f.v_act);
  EXPECT_GT(r.num_refreshes, 0);
  EXPECT_TRUE(r.x_nom.isApprox(f.openLoop(x0, f.v_nom), 1e-6f));
}

TEST(AdaptiveRMPPIReference, ZeroErrorNeverTriggersAndHasZeroFeedback)
{
  DubinsFixture f;
  f.config.enabled = true;
  f.config.distance_radius = 1e-4f;
  DubinsFixture::state_array x0;
  x0 << 0.1f, 0.2f, 0.3f;
  auto ref = f.make(x0);
  auto r = ref.rollout(0, x0, x0, f.v_nom, f.v_nom);
  EXPECT_TRUE(r.events.empty());
  EXPECT_NEAR(r.u_fb.cwiseAbs().maxCoeff(), 0.0f, 1e-7f);
  EXPECT_TRUE(r.x_act.isApprox(r.x_nom, 1e-6f));
}

TEST(AdaptiveRMPPIReference, ForcedTriggerRefreshesAtExpectedSampleAndTime)
{
  DubinsFixture f;
  f.config.enabled = true;
  f.config.distance_radius = 1.0f;
  f.config.max_refreshes_per_rollout = 1;
  DubinsFixture::state_array x0 = DubinsFixture::state_array::Zero();
  auto ref = f.make(x0);
  ref.setSolver(f.markerSolver(-3.0f));
  // Inject a perturbation that only exceeds the radius once the state has drifted: zero initial error, then a
  // large open-loop actual control at t = 4 drives the actual state away.
  DubinsFixture::control_traj v_act = f.v_nom;
  v_act(1, 4) += 30.0f;  // yaw rate impulse: yaw error 30 * dt = 1.5 rad at t = 5
  auto r_off_ref = f.make(x0);
  AdaptiveFeedbackConfig<3> off = f.config;
  off.enabled = false;
  r_off_ref.setConfig(off);
  auto r_off = r_off_ref.rollout(5, x0, x0, f.v_nom, v_act);
  auto r = ref.rollout(5, x0, x0, f.v_nom, v_act);
  ASSERT_GE(r.events.size(), 1u);
  // The marker policy tracks the nominal reference after its first step, so later triggers hit the budget.
  for (size_t i = 1; i < r.events.size(); i++)
  {
    EXPECT_EQ(r.events[i].outcome, RefreshOutcome::BUDGET_EXCEEDED);
  }
  EXPECT_LE(r.events.size(), 2u);
  EXPECT_EQ(r.events[0].sample, 5);
  EXPECT_EQ(r.events[0].t, 5);
  EXPECT_EQ(r.events[0].outcome, RefreshOutcome::REFRESHED);
  EXPECT_GT(r.events[0].distance, f.config.distance_radius);
  // Prefix identical, refreshed gain in use from t = 5 onward, successors differ.
  EXPECT_TRUE(r.x_act.leftCols(6).isApprox(r_off.x_act.leftCols(6), 1e-6f));
  DubinsFixture::control_array expected = (-3.0f * DubinsFixture::K_T::Identity()) * (r.x_act.col(5) - r.x_nom.col(5));
  EXPECT_TRUE(r.u_fb.col(5).isApprox(expected, 1e-6f));
  EXPECT_GT((r.x_act.col(7) - r_off.x_act.col(7)).norm(), 1e-4f);
}

TEST(AdaptiveRMPPIReference, FeedbackSignOpposesTrackingError)
{
  DIFixture f;
  f.config.enabled = false;
  DIFixture::state_array x0 = DIFixture::state_array::Zero();
  auto ref = f.make(x0);
  DIFixture::state_array x_act0;
  x_act0 << 1.0f, -2.0f, 0.0f, 0.0f;
  auto r = ref.rollout(0, x0, x_act0, f.v_nom, f.v_nom);
  // K = -0.5 I acting on the position block: feedback must point back toward x*.
  EXPECT_LT(r.u_fb(0, 0), 0.0f);
  EXPECT_GT(r.u_fb(1, 0), 0.0f);
}

TEST(AdaptiveRMPPIReference, SolverFailureKeepsPreviousGains)
{
  DubinsFixture f;
  f.config.enabled = true;
  f.config.distance_radius = 0.1f;
  f.config.max_refreshes_per_rollout = 100;
  DubinsFixture::state_array x0 = DubinsFixture::state_array::Zero();
  auto ref = f.make(x0);
  ref.setSolver([](const DubinsFixture::state_array&, int, const DubinsFixture::state_traj&,
                   const DubinsFixture::control_traj&, DubinsFixture::REF::Cache&) { return false; });
  DubinsFixture::state_array x_act0;
  x_act0 << 1.0f, 1.0f, 1.0f;
  auto r = ref.rollout(0, x0, x_act0, f.v_nom, f.v_act);
  ASSERT_FALSE(r.events.empty());
  for (const auto& e : r.events)
  {
    EXPECT_EQ(e.outcome, e.t < f.T - 1 ? RefreshOutcome::SOLVER_FAILED : RefreshOutcome::HORIZON_TOO_SHORT);
  }
  AdaptiveFeedbackConfig<3> off = f.config;
  off.enabled = false;
  ref.setConfig(off);
  auto r_off = ref.rollout(0, x0, x_act0, f.v_nom, f.v_act);
  EXPECT_TRUE(r.x_act.isApprox(r_off.x_act, 1e-6f));
}

TEST(AdaptiveRMPPIReference, NonFiniteSolverOutputIsRejected)
{
  DubinsFixture f;
  f.config.enabled = true;
  f.config.distance_radius = 0.1f;
  DubinsFixture::state_array x0 = DubinsFixture::state_array::Zero();
  auto ref = f.make(x0);
  ref.setSolver(f.markerSolver(std::numeric_limits<float>::quiet_NaN()));
  DubinsFixture::state_array x_act0;
  x_act0 << 1.0f, 1.0f, 1.0f;
  auto r = ref.rollout(0, x0, x_act0, f.v_nom, f.v_act);
  ASSERT_FALSE(r.events.empty());
  EXPECT_EQ(r.events[0].outcome, RefreshOutcome::SOLVER_FAILED);
  EXPECT_TRUE(r.x_act.allFinite());
}

TEST(AdaptiveRMPPIReference, PerRolloutAndPerIterationBudgetsAreEnforcedAndLogged)
{
  DubinsFixture f;
  f.config.enabled = true;
  f.config.distance_radius = -1.0f;  // always fires
  f.config.max_refreshes_per_rollout = 2;
  f.config.max_refreshes_per_iteration = 3;
  DubinsFixture::state_array x0 = DubinsFixture::state_array::Zero();
  auto ref = f.make(x0);
  int calls = 0;
  ref.setSolver(f.markerSolver(-1.0f, &calls));
  ref.resetIterationBudget();
  auto r0 = ref.rollout(0, x0, x0, f.v_nom, f.v_act);
  auto r1 = ref.rollout(1, x0, x0, f.v_nom, f.v_act);
  EXPECT_EQ(r0.num_refreshes, 2);
  EXPECT_EQ(r1.num_refreshes, 1);
  EXPECT_EQ(calls, 3);
  EXPECT_EQ(ref.getIterationRefreshes(), 3);
  // One overflow event per rollout, after the successful refreshes.
  ASSERT_EQ(r0.events.size(), 3u);
  EXPECT_EQ(r0.events.back().outcome, RefreshOutcome::BUDGET_EXCEEDED);
  ASSERT_EQ(r1.events.size(), 2u);
  EXPECT_EQ(r1.events.back().outcome, RefreshOutcome::BUDGET_EXCEEDED);
  ref.resetIterationBudget();
  EXPECT_EQ(ref.getIterationRefreshes(), 0);
}

TEST(AdaptiveRMPPIReference, NoRefreshWithLessThanTwoStepsRemaining)
{
  DubinsFixture f;
  f.config.enabled = true;
  f.config.distance_radius = 1.0f;
  DubinsFixture::state_array x0 = DubinsFixture::state_array::Zero();
  auto ref = f.make(x0);
  int calls = 0;
  ref.setSolver(f.markerSolver(-1.0f, &calls));
  DubinsFixture::control_traj v_act = f.v_nom;
  v_act(1, f.T - 2) += 40.0f;  // error appears only at the final timestep
  auto r = ref.rollout(0, x0, x0, f.v_nom, v_act);
  ASSERT_EQ(r.events.size(), 1u);
  EXPECT_EQ(r.events[0].t, f.T - 1);
  EXPECT_EQ(r.events[0].outcome, RefreshOutcome::HORIZON_TOO_SHORT);
  EXPECT_EQ(calls, 0);
}

TEST(AdaptiveRMPPIReference, SaturationClipsAppliedControlButNotFeedbackTerm)
{
  DIFixture f;
  for (int i = 0; i < DIFixture::C; i++)
  {
    f.model.control_rngs_[i] = make_float2(-0.2f, 0.2f);
  }
  f.config.enabled = false;
  DIFixture::state_array x0 = DIFixture::state_array::Zero();
  auto ref = f.make(x0);
  DIFixture::state_array x_act0;
  x_act0 << 10.0f, 10.0f, 0.0f, 0.0f;
  auto r = ref.rollout(0, x0, x_act0, f.v_nom, f.v_nom);
  EXPECT_LE(r.u_applied.cwiseAbs().maxCoeff(), 0.2f + 1e-6f);
  EXPECT_GT(r.u_fb.cwiseAbs().maxCoeff(), 1.0f);
}

TEST(AdaptiveRMPPIReference, ResidualTriggerIgnoresOffsetsOnLinearDynamics)
{
  DIFixture f;
  f.config.enabled = true;
  f.config.trigger = TriggerType::MODEL_RESIDUAL;
  f.config.residual_threshold = 1e-3f;
  f.config.hard_distance_radius = 100.0f;
  DIFixture::state_array x0 = DIFixture::state_array::Zero();
  auto ref = f.make(x0);
  DIFixture::state_array x_act0;
  x_act0 << 3.0f, -3.0f, 1.0f, 1.0f;
  auto r = ref.rollout(0, x0, x_act0, f.v_nom, f.v_act);
  // Linear plant: the affine model is exact, so a large but well-modelled offset is not a violation.
  EXPECT_TRUE(r.events.empty());

  AdaptiveFeedbackConfig<4> dist = f.config;
  dist.trigger = TriggerType::DISTANCE;
  dist.distance_radius = 1.0f;
  ref.setConfig(dist);
  ref.setSolver(f.markerSolver(-0.5f));
  EXPECT_FALSE(ref.rollout(0, x0, x_act0, f.v_nom, f.v_act).events.empty());
}

TEST(AdaptiveRMPPIReference, SwitchingToResidualTriggerBuildsBaselineJacobians)
{
  DIFixture f;
  f.config.enabled = true;
  f.config.trigger = TriggerType::DISTANCE;
  f.config.hard_distance_radius = 100.0f;
  DIFixture::state_array x0 = DIFixture::state_array::Zero();
  auto ref = f.make(x0);
  AdaptiveFeedbackConfig<4> residual = f.config;
  residual.trigger = TriggerType::MODEL_RESIDUAL;
  residual.residual_threshold = 1e-3f;
  ref.setConfig(residual);
  DIFixture::state_array x_act0;
  x_act0 << 3.0f, -3.0f, 1.0f, 1.0f;
  // Without rebuilt Jacobians this indexes empty A_d/B_d; with them the exact linear model never triggers.
  EXPECT_TRUE(ref.rollout(0, x0, x_act0, f.v_nom, f.v_act).events.empty());
}

TEST(AdaptiveRMPPIReference, ResidualTriggerFiresOnNonlinearDeviation)
{
  DubinsFixture f;
  f.config.enabled = true;
  f.config.trigger = TriggerType::MODEL_RESIDUAL;
  f.config.residual_threshold = 1e-3f;
  f.config.hard_distance_radius = 100.0f;
  f.config.angle_state_mask[2] = 1;
  f.v_nom = DubinsFixture::control_traj::Constant(2, f.T, 1.0f);
  DubinsFixture::state_array x0 = DubinsFixture::state_array::Zero();
  auto ref = f.make(x0);
  ref.setSolver(f.markerSolver(-0.5f));
  DubinsFixture::state_array x_act0;
  x_act0 << 0.0f, 0.0f, 2.0f;  // large heading error: linearization about x~ is poor
  auto r = ref.rollout(0, x0, x_act0, f.v_nom, f.v_nom);
  ASSERT_FALSE(r.events.empty());
  EXPECT_EQ(r.events[0].t, 1);
  EXPECT_GT(r.events[0].residual, f.config.residual_threshold);
}

TEST(AdaptiveRMPPIReference, DefaultDDPRefreshProducesFiniteLocalPolicy)
{
  DubinsFixture f;
  f.config.enabled = true;
  f.config.distance_radius = 0.5f;
  f.config.max_refreshes_per_rollout = 1;
  f.v_nom = DubinsFixture::control_traj::Constant(2, f.T, 1.0f);
  DubinsFixture::state_array x0 = DubinsFixture::state_array::Zero();
  auto ref = f.make(x0);
  DubinsFixture::state_array x_act0;
  x_act0 << 0.0f, 1.0f, 0.5f;
  auto r = ref.rollout(0, x0, x_act0, f.v_nom, f.v_nom);
  ASSERT_EQ(r.events.size(), 1u);
  EXPECT_EQ(r.events[0].t, 0);
  EXPECT_EQ(r.events[0].outcome, RefreshOutcome::REFRESHED);
  EXPECT_TRUE(r.x_act.allFinite());
  EXPECT_TRUE(r.u_fb.allFinite());
  // Direct check of the cached recovery trajectory: it starts at the actual state it was solved from.
  DubinsFixture::REF::Cache c = ref.getBaselineCache();
  ASSERT_EQ(ref.refresh(c, 3, x_act0), RefreshOutcome::REFRESHED);
  EXPECT_EQ(c.start_t, 3);
  EXPECT_EQ(c.size(), f.T - 3);
  EXPECT_TRUE(c.x_ref[0].isApprox(x_act0, 1e-6f));
}
