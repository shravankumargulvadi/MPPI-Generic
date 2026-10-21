#include <mppi/controllers/R-MPPI/adaptive_robust_mppi_controller.cuh>

#define ADAPTIVE_RMPPI_TEMPLATE                                                                                        \
  template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, class SAMPLING_T,              \
            class PARAMS_T>
#define AdaptiveRMPPI AdaptiveRobustMPPIController<DYN_T, COST_T, FB_T, MAX_TIMESTEPS, NUM_ROLLOUTS, SAMPLING_T, PARAMS_T>

ADAPTIVE_RMPPI_TEMPLATE
AdaptiveRMPPI::~AdaptiveRobustMPPIController()
{
  if (saved_samples_d_)
  {
    cudaFree(saved_samples_d_);
    saved_samples_d_ = nullptr;
  }
}

ADAPTIVE_RMPPI_TEMPLATE
void AdaptiveRMPPI::ensureCombinedKernel()
{
  if (shared_mem_per_block_ == 0)
  {
    cudaDeviceProp device_prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&device_prop, 0));
    shared_mem_per_block_ = device_prop.sharedMemPerBlock;
  }
  const size_t combined_bytes = mppi::kernels::rmppi::calcRMPPICombinedKernelSharedMemSize(
      this->model_, this->cost_, this->sampler_, this->fb_controller_->getHostPointer().get(),
      this->params_.dynamics_rollout_dim_);
  if (combined_bytes > shared_mem_per_block_)
  {
    throw std::runtime_error("Adaptive RMPPI requires the combined rollout kernel, which needs " +
                             std::to_string(combined_bytes) + " bytes of shared memory; the device allows " +
                             std::to_string(shared_mem_per_block_) +
                             ". Lower dynamics_rollout_dim_ or disable adaptive feedback.");
  }
  // Keep the inherited sampled-trajectory/visualization paths consistent with the kernel actually used.
  if (this->getKernelChoiceAsEnum() != kernelType::USE_SINGLE_KERNEL)
  {
    this->setKernelChoice(kernelType::USE_SINGLE_KERNEL);
  }
}

ADAPTIVE_RMPPI_TEMPLATE
void AdaptiveRMPPI::launchSingleKernel()
{
  mppi::kernels::rmppi::launchRMPPIRolloutKernel<DYN_T, COST_T, SAMPLING_T, FEEDBACK_GPU>(
      this->model_, this->cost_, this->sampler_, this->fb_controller_->getHostPointer().get(), this->getDt(),
      this->getNumTimesteps(), NUM_ROLLOUTS, this->getLambda(), this->getAlpha(), this->getValueFunctionThreshold(),
      this->initial_state_d_, this->trajectory_costs_d_, this->params_.dynamics_rollout_dim_, this->stream_, false);
}

ADAPTIVE_RMPPI_TEMPLATE
void AdaptiveRMPPI::copyPolicyAssignmentToDevice(const std::vector<int>& policy_id,
                                                 const std::vector<int>& policy_start)
{
  auto* s = this->fb_controller_->getFeedbackStatePointer();
  HANDLE_ERROR(cudaMemcpyAsync(s->policy_id_d_, policy_id.data(), sizeof(int) * NUM_ROLLOUTS, cudaMemcpyHostToDevice,
                               this->stream_));
  HANDLE_ERROR(cudaMemcpyAsync(s->policy_start_t_d_, policy_start.data(), sizeof(int) * NUM_ROLLOUTS,
                               cudaMemcpyHostToDevice, this->stream_));
}

ADAPTIVE_RMPPI_TEMPLATE
void AdaptiveRMPPI::launchRolloutKernels()
{
  using mppi::adaptive::RefreshEvent;
  using mppi::adaptive::RefreshOutcome;
  const Config& cfg = this->fb_controller_->getAdaptiveConfig();
  if (!cfg.enabled)
  {
    last_stats_ = Stats();
    PARENT_CLASS::launchRolloutKernels();
    return;
  }
  ensureCombinedKernel();
  constexpr int S_DIM = DYN_T::STATE_DIM;
  auto gpu_fb = this->fb_controller_->getHostPointer();
  gpu_fb->setNumRollouts(NUM_ROLLOUTS);
  auto* fb_state = this->fb_controller_->getFeedbackStatePointer();

  Stats stats;
  stats.iterations = 1;
  std::vector<int> policy_id(NUM_ROLLOUTS, -1), policy_start(NUM_ROLLOUTS, 0), attempts(NUM_ROLLOUTS, 0);
  std::vector<int> violation_t(NUM_ROLLOUTS, -1);
  std::vector<float> violation_x(NUM_ROLLOUTS * S_DIM, 0.0f);
  copyPolicyAssignmentToDevice(policy_id, policy_start);

  // The rollout kernel overwrites the sampled controls with the applied (feedback + constrained) controls, so the
  // original noise must be restored before a replay.
  const size_t num_sample_floats = static_cast<size_t>(this->sampler_->getNumDistributions()) * NUM_ROLLOUTS *
                                   this->getNumTimesteps() * DYN_T::CONTROL_DIM;
  if (saved_samples_size_ != num_sample_floats)
  {
    if (saved_samples_d_)
    {
      HANDLE_ERROR(cudaFree(saved_samples_d_));
    }
    HANDLE_ERROR(cudaMalloc((void**)&saved_samples_d_, sizeof(float) * num_sample_floats));
    saved_samples_size_ = num_sample_floats;
  }
  float* samples_d = this->sampler_->getControlSample(0, 0, 0);
  HANDLE_ERROR(cudaMemcpyAsync(saved_samples_d_, samples_d, sizeof(float) * num_sample_floats,
                               cudaMemcpyDeviceToDevice, this->stream_));

  int slots_used = 0;
  int iteration_refreshes = 0;
  const int max_passes = std::max(cfg.max_refreshes_per_rollout, 0) + 1;
  for (int pass = 0; pass < max_passes; pass++)
  {
    if (pass > 0)
    {
      HANDLE_ERROR(cudaMemcpyAsync(samples_d, saved_samples_d_, sizeof(float) * num_sample_floats,
                                   cudaMemcpyDeviceToDevice, this->stream_));
    }
    // Recording never changes the applied control, so the last pass is recorded too for diagnostics; its
    // violations cannot be refreshed and are reported as budget overflows.
    const bool last_pass = pass == max_passes - 1;
    fb_state->record_violations_ = 1;
    gpu_fb->copyFlagsToDevice(false);
    HANDLE_ERROR(cudaMemsetAsync(fb_state->violation_t_d_, 0xFF, sizeof(int) * NUM_ROLLOUTS, this->stream_));
    launchSingleKernel();
    stats.passes++;

    HANDLE_ERROR(cudaMemcpyAsync(violation_t.data(), fb_state->violation_t_d_, sizeof(int) * NUM_ROLLOUTS,
                                 cudaMemcpyDeviceToHost, this->stream_));
    HANDLE_ERROR(cudaMemcpyAsync(violation_x.data(), fb_state->violation_x_d_, sizeof(float) * NUM_ROLLOUTS * S_DIM,
                                 cudaMemcpyDeviceToHost, this->stream_));
    HANDLE_ERROR(cudaStreamSynchronize(this->stream_));

    bool policies_changed = false;
    for (int n = 0; n < NUM_ROLLOUTS; n++)
    {
      const int t = violation_t[n];
      if (t < 0)
      {
        continue;
      }
      stats.violations++;
      Eigen::Map<const typename DYN_T::state_array> x(&violation_x[n * S_DIM]);
      const bool on_policy = policy_id[n] >= 0 && t >= policy_start[n];
      const float* x_ref = on_policy ?
                               &fb_state->policy_x_ref_[policy_id[n] * FB_T::INTERNAL_STATE_T::REF_SIZE + S_DIM * t] :
                               &fb_state->x_ref_traj_[S_DIM * t];
      RefreshEvent e;
      e.sample = n;
      e.t = t;
      e.distance = mppi::adaptive::weightedStateDistance<S_DIM>(x.data(), x_ref, cfg.state_weights,
                                                                 cfg.angle_state_mask);
      if (last_pass || attempts[n] >= cfg.max_refreshes_per_rollout || iteration_refreshes >= cfg.max_refreshes_per_iteration ||
          slots_used >= FB_T::MAX_POLICY_SLOTS)
      {
        e.outcome = RefreshOutcome::BUDGET_EXCEEDED;
        stats.budget_overflows++;
      }
      else
      {
        attempts[n]++;
        iteration_refreshes++;
        e.outcome = this->fb_controller_->refreshPolicy(slots_used, t, x, policy_id[n]);
        if (e.outcome == RefreshOutcome::REFRESHED)
        {
          // The new slot carries the earlier segments forward, so the earliest start is kept.
          if (policy_id[n] < 0)
          {
            policy_start[n] = t;
          }
          policy_id[n] = slots_used++;
          policies_changed = true;
          stats.refreshes++;
        }
        else if (e.outcome == RefreshOutcome::SOLVER_FAILED)
        {
          stats.solver_failures++;
          this->logger_->warning("Adaptive RMPPI: gain refresh failed for sample %d at t = %d, keeping previous "
                                 "gains\n", n, t);
        }
        else
        {
          stats.horizon_too_short++;
        }
      }
      stats.events.push_back(e);
    }
    if (last_pass || !policies_changed)
    {
      // Nothing to replay: the costs of this pass already correspond to the policies that were used.
      break;
    }
    copyPolicyAssignmentToDevice(policy_id, policy_start);
  }
  fb_state->record_violations_ = 0;
  gpu_fb->copyFlagsToDevice(false);
  if (stats.budget_overflows > 0)
  {
    this->logger_->debug("Adaptive RMPPI: %d refresh requests exceeded the budget this iteration\n",
                         stats.budget_overflows);
  }
  last_stats_ = stats;
  total_stats_.accumulate(stats);
}

#undef AdaptiveRMPPI
#undef ADAPTIVE_RMPPI_TEMPLATE
