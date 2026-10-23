#include "/repo/tests/controllers/adaptive_rmppi_test.cu"

TEST(AdaptiveReview, DisabledParityWithPinnedEvaluationKernels)
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
  std::cout << "autotuned eval choices " << stock.getEvalKernelChoiceAsInt() << " "
            << adaptive.getEvalKernelChoiceAsInt() << std::endl;
  stock.setEvalKernelChoice(kernelType::USE_SINGLE_KERNEL);
  adaptive.setEvalKernelChoice(kernelType::USE_SINGLE_KERNEL);
  auto us = runClosedLoop(stock, a.model, a.dt, 10);
  auto ua = runClosedLoop(adaptive, b.model, b.dt, 10);
  for (size_t i = 0; i < us.size(); ++i)
  {
    std::cout << "cycle " << i << " stock " << us[i].transpose() << " adaptive " << ua[i].transpose()
              << " max_abs_diff " << (ua[i] - us[i]).cwiseAbs().maxCoeff() << std::endl;
    EXPECT_TRUE(ua[i].isApprox(us[i], 1e-4f));
  }
}

class ReplayUnderTest : public ADAPTIVE_BASE
{
public:
  using ADAPTIVE_BASE::ADAPTIVE_BASE;
  void replay(const DYN::state_array& nominal, const DYN::state_array& actual)
  {
    HANDLE_ERROR(cudaMemcpy(this->initial_state_d_, nominal.data(), sizeof(float) * DYN::STATE_DIM,
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(this->initial_state_d_ + DYN::STATE_DIM, actual.data(),
                            sizeof(float) * DYN::STATE_DIM, cudaMemcpyHostToDevice));
    this->fb_controller_->copyToDevice(true);
    this->launchRolloutKernels();
    HANDLE_ERROR(cudaDeviceSynchronize());
  }
};

TEST(AdaptiveReview, SecondRefreshPreservesFirstRefreshPrefix)
{
  Rig rig(1.0f);
  ADAPTIVE_FB fb(&rig.model, rig.dt);
  setFbParams(fb);
  ReplayUnderTest c(&rig.model, &rig.cost, &fb, &rig.sampler, rig.dt, 1, 4.0f, 0.0f, 10.0f);
  configure(c, 42);
  ADAPTIVE_FB::Config cfg;
  cfg.enabled = true;
  cfg.distance_radius = 0.001f;
  cfg.max_refreshes_per_rollout = 2;
  cfg.max_refreshes_per_iteration = 2;
  c.setAdaptiveConfig(cfg);
  using Ref = ADAPTIVE_FB::Reference;
  Ref::state_traj xref = Ref::state_traj::Zero(DYN::STATE_DIM, T);
  Ref::control_traj vnom = Ref::control_traj::Zero(DYN::CONTROL_DIM, T), vact = vnom;
  vact(0, 0) = 10.0f;
  std::vector<Ref::feedback_matrix> gains(T, Ref::feedback_matrix::Zero());
  auto& ref = fb.getReference();
  ref.setBaselineCache(gains, xref, vnom);
  ref.setRefreshReference(xref, vnom);
  ref.setSolver([](const DYN::state_array& x, int t0, const Ref::state_traj& xg,
                   const Ref::control_traj& ug, Ref::Cache& out) {
    out = Ref::Cache();
    out.start_t = t0;
    for (int k = 0; k < xg.cols(); ++k)
    {
      auto K = Ref::feedback_matrix::Zero().eval();
      K(0, 2) = -static_cast<float>(t0);
      K(1, 3) = -static_cast<float>(t0);
      out.K.push_back(K);
      out.x_ref.push_back(x);
      out.u_ref.push_back(ug.col(k));
    }
    return true;
  });
  auto* s = fb.getFeedbackStatePointer();
  std::fill_n(s->fb_gain_traj_, ADAPTIVE_FB::INTERNAL_STATE_T::GAIN_SIZE, 0.0f);
  std::fill_n(s->x_ref_traj_, ADAPTIVE_FB::INTERNAL_STATE_T::REF_SIZE, 0.0f);
  DYN::state_array x0 = DYN::state_array::Zero();
  ref.resetIterationBudget();
  auto oracle = ref.rollout(0, x0, x0, vnom, vact);
  std::vector<float> samples(2 * N * T * DYN::CONTROL_DIM, 0.0f);
  samples[N * T * DYN::CONTROL_DIM] = 10.0f;
  HANDLE_ERROR(cudaMemcpy(rig.sampler.getControlSample(0, 0, 0), samples.data(), sizeof(float) * samples.size(),
                          cudaMemcpyHostToDevice));
  c.replay(x0, x0);
  DYN::control_array applied;
  HANDLE_ERROR(cudaMemcpy(applied.data(), rig.sampler.getControlSample(0, 1, 1),
                          sizeof(float) * DYN::CONTROL_DIM, cudaMemcpyDeviceToHost));
  for (auto e : c.getLastIterationStats().events)
    std::cout << "event sample=" << e.sample << " t=" << e.t << " outcome=" << static_cast<int>(e.outcome) << std::endl;
  ASSERT_EQ(c.getLastIterationStats().refreshes, 2);
  ASSERT_EQ(oracle.num_refreshes, 2);
  std::cout << "t=1 CPU control=" << oracle.u_applied.col(1).transpose()
            << " GPU control=" << applied.transpose() << std::endl;
  EXPECT_TRUE(applied.isApprox(oracle.u_applied.col(1), 1e-5f));
}

TEST(AdaptiveReview, StockMatchesStockWithPinnedKernels)
{
  Rig a(1.0f), b(1.0f);
  STOCK_FB fa(&a.model, a.dt), fb(&b.model, b.dt);
  setFbParams(fa);
  setFbParams(fb);
  STOCK ca(&a.model, &a.cost, &fa, &a.sampler, a.dt, 1, 4.0f, 0.0f, 10.0f);
  STOCK cb(&b.model, &b.cost, &fb, &b.sampler, b.dt, 1, 4.0f, 0.0f, 10.0f);
  configure(ca, 42);
  configure(cb, 42);
  ca.setEvalKernelChoice(kernelType::USE_SINGLE_KERNEL);
  cb.setEvalKernelChoice(kernelType::USE_SINGLE_KERNEL);
  auto ua = runClosedLoop(ca, a.model, a.dt, 10);
  auto ub = runClosedLoop(cb, b.model, b.dt, 10);
  for (size_t i = 0; i < ua.size(); ++i)
  {
    std::cout << "stock/stock cycle " << i << " max_abs_diff " << (ua[i]-ub[i]).cwiseAbs().maxCoeff() << std::endl;
    EXPECT_TRUE(ua[i].isApprox(ub[i], 1e-4f));
  }
}
