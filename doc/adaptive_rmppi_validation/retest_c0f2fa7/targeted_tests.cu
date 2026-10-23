#include "/repo/doc/adaptive_rmppi_validation/review_diagnostics.cu"

class RetestUnderTest : public ReplayUnderTest
{
public:
  using ReplayUnderTest::ReplayUnderTest;
  void requireCombined() { this->ensureCombinedKernel(); }
};

TEST(AdaptiveRetest, GPURejectsEnabledResidualTriggerWithoutChangingConfig)
{
  Rig rig(1.0f);
  ADAPTIVE_FB fb(&rig.model, rig.dt);
  ADAPTIVE_FB::Config cfg;
  cfg.enabled = true;
  cfg.trigger = mppi::adaptive::TriggerType::MODEL_RESIDUAL;
  EXPECT_THROW(fb.setAdaptiveConfig(cfg), std::invalid_argument);
  EXPECT_FALSE(fb.getAdaptiveConfig().enabled);
  EXPECT_EQ(fb.getAdaptiveConfig().trigger, mppi::adaptive::TriggerType::DISTANCE);
  cfg.enabled = false;
  EXPECT_NO_THROW(fb.setAdaptiveConfig(cfg));
  cfg.trigger = mppi::adaptive::TriggerType::DISTANCE;
  cfg.enabled = true;
  EXPECT_NO_THROW(fb.setAdaptiveConfig(cfg));
}

TEST(AdaptiveRetest, ZeroRefreshBudgetStillReportsViolations)
{
  Rig rig(1.0f);
  ADAPTIVE_FB fb(&rig.model, rig.dt);
  setFbParams(fb);
  ReplayUnderTest c(&rig.model, &rig.cost, &fb, &rig.sampler, rig.dt, 1, 4.0f, 0.0f, 10.0f);
  configure(c, 42);
  ADAPTIVE_FB::Config cfg;
  cfg.enabled = true;
  cfg.distance_radius = .001f;
  cfg.max_refreshes_per_rollout = 0;
  cfg.max_refreshes_per_iteration = 0;
  c.setAdaptiveConfig(cfg);
  std::vector<float> samples(2*N*T*DYN::CONTROL_DIM, 0.0f);
  HANDLE_ERROR(cudaMemcpy(rig.sampler.getControlSample(0,0,0), samples.data(), samples.size()*sizeof(float), cudaMemcpyHostToDevice));
  DYN::state_array nominal = DYN::state_array::Zero(), actual = nominal;
  actual(0) = 1;
  c.replay(nominal, actual);
  const auto& s = c.getLastIterationStats();
  EXPECT_EQ(s.passes, 1);
  EXPECT_EQ(s.refreshes, 0);
  EXPECT_EQ(s.violations, N);
  EXPECT_EQ(s.budget_overflows, N);
}

TEST(AdaptiveRetest, FinalReplayReportsUnmetRefreshDemand)
{
  Rig rig(1.0f);
  ADAPTIVE_FB fb(&rig.model, rig.dt);
  setFbParams(fb);
  ReplayUnderTest c(&rig.model, &rig.cost, &fb, &rig.sampler, rig.dt, 1, 4.0f, 0.0f, 10.0f);
  configure(c, 42);
  ADAPTIVE_FB::Config cfg;
  cfg.enabled = true;
  cfg.distance_radius = .001f;
  cfg.max_refreshes_per_rollout = 1;
  cfg.max_refreshes_per_iteration = 1;
  c.setAdaptiveConfig(cfg);
  using Ref = ADAPTIVE_FB::Reference;
  Ref::state_traj xref = Ref::state_traj::Zero(DYN::STATE_DIM,T);
  Ref::control_traj u = Ref::control_traj::Zero(DYN::CONTROL_DIM,T);
  auto& ref = fb.getReference();
  std::vector<Ref::feedback_matrix> gains(T, Ref::feedback_matrix::Zero());
  ref.setBaselineCache(gains, xref, u);
  ref.setRefreshReference(xref, u);
  ref.setSolver([](const DYN::state_array& x, int t0, const Ref::state_traj& xg, const Ref::control_traj& ug, Ref::Cache& out) {
    out = Ref::Cache(); out.start_t = t0;
    for (int k=0; k<xg.cols(); ++k) {
      out.K.push_back(Ref::feedback_matrix::Zero());
      out.x_ref.push_back(x);
      out.u_ref.push_back(ug.col(k));
    }
    return true;
  });
  auto* state = fb.getFeedbackStatePointer();
  std::fill_n(state->fb_gain_traj_, ADAPTIVE_FB::INTERNAL_STATE_T::GAIN_SIZE,0.0f);
  std::fill_n(state->x_ref_traj_, ADAPTIVE_FB::INTERNAL_STATE_T::REF_SIZE,0.0f);
  std::vector<float> samples(2*N*T*DYN::CONTROL_DIM,0.0f);
  samples[N*T*DYN::CONTROL_DIM] = 10;
  HANDLE_ERROR(cudaMemcpy(rig.sampler.getControlSample(0,0,0),samples.data(),samples.size()*sizeof(float),cudaMemcpyHostToDevice));
  DYN::state_array x0 = DYN::state_array::Zero();
  c.replay(x0,x0);
  const auto& s = c.getLastIterationStats();
  EXPECT_EQ(s.passes,2);
  EXPECT_EQ(s.refreshes,1);
  EXPECT_EQ(s.budget_overflows,1);
  ASSERT_EQ(s.events.size(),2u);
  EXPECT_EQ(s.events[0].t,1);
  EXPECT_EQ(s.events[1].t,2);
  EXPECT_EQ(s.events[1].outcome,mppi::adaptive::RefreshOutcome::BUDGET_EXCEEDED);
}

TEST(AdaptiveRetest, DisablingClearsLastIterationStats)
{
  Rig rig(1.0f);
  ADAPTIVE_FB fb(&rig.model,rig.dt);
  setFbParams(fb);
  AdaptiveUnderTest c(&rig.model,&rig.cost,&fb,&rig.sampler,rig.dt,1,4.0f,0.0f,10.0f);
  configure(c,42);
  ADAPTIVE_FB::Config cfg;
  cfg.enabled=true; cfg.distance_radius=.001f;
  c.setAdaptiveConfig(cfg);
  runClosedLoop(c,rig.model,rig.dt,1);
  EXPECT_GT(c.getLastIterationStats().passes,0);
  cfg.enabled=false;
  c.setAdaptiveConfig(cfg);
  runClosedLoop(c,rig.model,rig.dt,1);
  EXPECT_EQ(c.getLastIterationStats().passes,0);
  EXPECT_EQ(c.getLastIterationStats().violations,0);
  EXPECT_TRUE(c.getLastIterationStats().events.empty());
}

TEST(AdaptiveRetest, EnabledModeSelectsCombinedKernel)
{
  Rig rig(1.0f);
  ADAPTIVE_FB fb(&rig.model,rig.dt);
  setFbParams(fb);
  AdaptiveUnderTest c(&rig.model,&rig.cost,&fb,&rig.sampler,rig.dt,1,4.0f,0.0f,10.0f);
  configure(c,42);
  c.setKernelChoice(kernelType::USE_SPLIT_KERNELS);
  ADAPTIVE_FB::Config cfg;
  cfg.enabled=true; cfg.distance_radius=1e9f;
  c.setAdaptiveConfig(cfg);
  runClosedLoop(c,rig.model,rig.dt,1);
  EXPECT_EQ(c.getKernelChoiceAsEnum(),kernelType::USE_SINGLE_KERNEL);
}

TEST(AdaptiveRetest, CombinedKernelRejectsExcessSharedMemory)
{
  Rig rig(1.0f);
  ADAPTIVE_FB fb(&rig.model,rig.dt);
  setFbParams(fb);
  RetestUnderTest c(&rig.model,&rig.cost,&fb,&rig.sampler,rig.dt,1,4.0f,0.0f,10.0f);
  configure(c,42);
  auto p=c.getParams();
  p.dynamics_rollout_dim_=dim3(1024,4,2);
  c.setParams(p);
  EXPECT_THROW(c.requireCombined(),std::runtime_error);
}
