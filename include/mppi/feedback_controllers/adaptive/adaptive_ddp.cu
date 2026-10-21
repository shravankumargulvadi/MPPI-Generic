#include <mppi/feedback_controllers/adaptive/adaptive_ddp.cuh>

#include <cstring>

#define ADAPTIVE_DDP_DEVICE_TEMPLATE template <class GPU_FB_T, class DYN_T, int NUM_TIMESTEPS, int MAX_POLICIES>
#define AdaptiveDDPDevice DeviceAdaptiveDDPImpl<GPU_FB_T, DYN_T, NUM_TIMESTEPS, MAX_POLICIES>

ADAPTIVE_DDP_DEVICE_TEMPLATE
void AdaptiveDDPDevice::setNumRollouts(int num_rollouts)
{
  if (num_rollouts == num_rollouts_)
  {
    return;
  }
  if (this->GPUMemStatus_)
  {
    deallocateCUDAMemory();
  }
  num_rollouts_ = num_rollouts;
  this->state_.num_rollouts_ = num_rollouts;
  if (this->GPUMemStatus_)
  {
    allocateRolloutBuffers();
    this->copyToDevice(true);
  }
}

ADAPTIVE_DDP_DEVICE_TEMPLATE
void AdaptiveDDPDevice::allocateRolloutBuffers()
{
  if (num_rollouts_ <= 0)
  {
    return;
  }
  auto& s = this->state_;
  HANDLE_ERROR(cudaMalloc((void**)&s.policy_id_d_, sizeof(int) * num_rollouts_));
  HANDLE_ERROR(cudaMalloc((void**)&s.policy_start_t_d_, sizeof(int) * num_rollouts_));
  HANDLE_ERROR(cudaMalloc((void**)&s.violation_t_d_, sizeof(int) * num_rollouts_));
  HANDLE_ERROR(cudaMalloc((void**)&s.violation_x_d_, sizeof(float) * DYN_T::STATE_DIM * num_rollouts_));
  // 0xFF bytes == -1 for int: every rollout starts on the baseline policy with no recorded violation.
  HANDLE_ERROR(cudaMemset(s.policy_id_d_, 0xFF, sizeof(int) * num_rollouts_));
  HANDLE_ERROR(cudaMemset(s.policy_start_t_d_, 0, sizeof(int) * num_rollouts_));
  HANDLE_ERROR(cudaMemset(s.violation_t_d_, 0xFF, sizeof(int) * num_rollouts_));
}

ADAPTIVE_DDP_DEVICE_TEMPLATE
void AdaptiveDDPDevice::deallocateCUDAMemory()
{
  auto& s = this->state_;
  cudaFree(s.policy_id_d_);
  cudaFree(s.policy_start_t_d_);
  cudaFree(s.violation_t_d_);
  cudaFree(s.violation_x_d_);
  s.policy_id_d_ = nullptr;
  s.policy_start_t_d_ = nullptr;
  s.violation_t_d_ = nullptr;
  s.violation_x_d_ = nullptr;
}

ADAPTIVE_DDP_DEVICE_TEMPLATE
void AdaptiveDDPDevice::copyPolicyToDevice(int slot, bool synchronize)
{
  if (!this->GPUMemStatus_)
  {
    return;
  }
  const int g = STATE_T::GAIN_SIZE, r = STATE_T::REF_SIZE;
  HANDLE_ERROR(cudaMemcpyAsync(&this->feedback_d_->state_.policy_gains_[slot * g], &this->state_.policy_gains_[slot * g],
                               sizeof(float) * g, cudaMemcpyHostToDevice, this->stream_));
  HANDLE_ERROR(cudaMemcpyAsync(&this->feedback_d_->state_.policy_x_ref_[slot * r], &this->state_.policy_x_ref_[slot * r],
                               sizeof(float) * r, cudaMemcpyHostToDevice, this->stream_));
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(this->stream_));
  }
}

ADAPTIVE_DDP_DEVICE_TEMPLATE
void AdaptiveDDPDevice::copyFlagsToDevice(bool synchronize)
{
  if (!this->GPUMemStatus_)
  {
    return;
  }
  auto* d = &this->feedback_d_->state_;
  auto* h = &this->state_;
  HANDLE_ERROR(cudaMemcpyAsync(&d->state_weights_, &h->state_weights_, sizeof(h->state_weights_),
                               cudaMemcpyHostToDevice, this->stream_));
  HANDLE_ERROR(cudaMemcpyAsync(&d->angle_mask_, &h->angle_mask_, sizeof(h->angle_mask_), cudaMemcpyHostToDevice,
                               this->stream_));
  HANDLE_ERROR(cudaMemcpyAsync(&d->distance_radius_, &h->distance_radius_, sizeof(float), cudaMemcpyHostToDevice,
                               this->stream_));
  HANDLE_ERROR(cudaMemcpyAsync(&d->adaptive_enabled_, &h->adaptive_enabled_, sizeof(int), cudaMemcpyHostToDevice,
                               this->stream_));
  HANDLE_ERROR(cudaMemcpyAsync(&d->record_violations_, &h->record_violations_, sizeof(int), cudaMemcpyHostToDevice,
                               this->stream_));
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(this->stream_));
  }
}

ADAPTIVE_DDP_DEVICE_TEMPLATE
__device__ void AdaptiveDDPDevice::k(const float* __restrict__ x_act, const float* __restrict__ x_goal, const int t,
                                     float* __restrict__ theta, float* __restrict__ control_output)
{
  const STATE_T& s = this->state_;
  const float* fb_gain_t = &(s.fb_gain_traj_[DYN_T::STATE_DIM * DYN_T::CONTROL_DIM * t]);
  if (s.adaptive_enabled_)
  {
    // Valid in kernels where threadIdx.x/blockIdx.x enumerate samples (rolloutRMPPIKernel).
    const int sample = blockDim.x * blockIdx.x + threadIdx.x;
    if (sample < s.num_rollouts_)
    {
      const float* x_ref_t = &s.x_ref_traj_[DYN_T::STATE_DIM * t];
      const int pid = s.policy_id_d_[sample];
      const int start = s.policy_start_t_d_[sample];
      const bool use_policy = pid >= 0 && t >= start;
      if (use_policy)
      {
        fb_gain_t = &s.policy_gains_[pid * STATE_T::GAIN_SIZE + DYN_T::STATE_DIM * DYN_T::CONTROL_DIM * t];
        x_ref_t = &s.policy_x_ref_[pid * STATE_T::REF_SIZE + DYN_T::STATE_DIM * t];
      }
      if (s.record_violations_ && (!use_policy || t > start) && s.violation_t_d_[sample] < 0)
      {
        const float d = mppi::adaptive::weightedStateDistance<DYN_T::STATE_DIM>(x_act, x_ref_t, s.state_weights_,
                                                                                s.angle_mask_);
        // Single writer per rollout; other threadIdx.y lanes only read, which cannot change the control.
        if (d > s.distance_radius_ && threadIdx.y == 0)
        {
          s.violation_t_d_[sample] = t;
          for (int i = 0; i < DYN_T::STATE_DIM; i++)
          {
            s.violation_x_d_[sample * DYN_T::STATE_DIM + i] = x_act[i];
          }
        }
      }
    }
  }
  // Tracking law identical to DeviceDDPImpl::k, with the gain pointer possibly swapped.
  for (int j = 0; j < DYN_T::CONTROL_DIM; j++)
  {
    control_output[j] = 0.0f;
  }
  float e = 0;
  for (int i = 0; i < DYN_T::STATE_DIM; i++)
  {
    e = x_act[i] - x_goal[i];
    if (DYN_T::CONTROL_DIM % 4 == 0)
    {
      const float4* fb_gain_t4 = reinterpret_cast<const float4*>(&fb_gain_t[i * DYN_T::CONTROL_DIM]);
      for (int j = 0; j < DYN_T::CONTROL_DIM / 4; j++)
      {
        reinterpret_cast<float4*>(control_output)[j] += fb_gain_t4[j] * e;
      }
    }
    else if (DYN_T::CONTROL_DIM % 2 == 0)
    {
      const float2* fb_gain_t2 = reinterpret_cast<const float2*>(&fb_gain_t[i * DYN_T::CONTROL_DIM]);
      for (int j = 0; j < DYN_T::CONTROL_DIM / 2; j++)
      {
        reinterpret_cast<float2*>(control_output)[j] += fb_gain_t2[j] * e;
      }
    }
    else
    {
      for (int j = 0; j < DYN_T::CONTROL_DIM; j++)
      {
        control_output[j] += fb_gain_t[i * DYN_T::CONTROL_DIM + j] * e;
      }
    }
  }
}

#undef AdaptiveDDPDevice
#undef ADAPTIVE_DDP_DEVICE_TEMPLATE

#define ADAPTIVE_DDP_TEMPLATE template <class DYN_T, int NUM_TIMESTEPS, int MAX_POLICIES>
#define AdaptiveDDP AdaptiveDDPFeedback<DYN_T, NUM_TIMESTEPS, MAX_POLICIES>

ADAPTIVE_DDP_TEMPLATE
AdaptiveDDP::AdaptiveDDPFeedback(DYN_T* model, float dt, int num_timesteps, cudaStream_t stream)
  : PARENT_CLASS(dt, std::max(num_timesteps, NUM_TIMESTEPS), stream), model_(model)
{
  baseline_ddp_ = std::make_shared<DDPFeedback<DYN_T, NUM_TIMESTEPS>>(model, dt, num_timesteps, stream);
  reference_ = std::make_shared<Reference>(model_, dt, this->num_timesteps_, config_, this->params_);
  setAdaptiveConfig(config_);
}

ADAPTIVE_DDP_TEMPLATE
void AdaptiveDDP::initTrackingController()
{
  baseline_ddp_->setParams(this->params_);
  baseline_ddp_->initTrackingController();
  reference_ = std::make_shared<Reference>(model_, this->dt_, this->num_timesteps_, config_, this->params_);
}

ADAPTIVE_DDP_TEMPLATE
void AdaptiveDDP::setParams(const DDPParams<DYN_T>& params)
{
  this->params_ = params;
  baseline_ddp_->setParams(params);
  reference_ = std::make_shared<Reference>(model_, this->dt_, this->num_timesteps_, config_, this->params_);
}

ADAPTIVE_DDP_TEMPLATE
void AdaptiveDDP::setAdaptiveConfig(const Config& config)
{
  config_ = config;
  reference_->setConfig(config);
  auto* s = this->getFeedbackStatePointer();
  for (int i = 0; i < DYN_T::STATE_DIM; i++)
  {
    s->state_weights_[i] = config.state_weights[i];
    s->angle_mask_[i] = config.angle_state_mask[i];
  }
  s->distance_radius_ = config.distance_radius;
  s->adaptive_enabled_ = config.enabled ? 1 : 0;
  this->gpu_controller_->copyFlagsToDevice(true);
}

ADAPTIVE_DDP_TEMPLATE
typename AdaptiveDDP::control_array AdaptiveDDP::k_(const Eigen::Ref<const state_array>& x_act,
                                                    const Eigen::Ref<const state_array>& x_goal, int t,
                                                    INTERNAL_STATE_T& fb_state)
{
  int index = DYN_T::STATE_DIM * DYN_T::CONTROL_DIM * t;
  Eigen::Map<feedback_gain_matrix> fb_gain(&(fb_state.fb_gain_traj_[index]));
  control_array u_output = fb_gain * (x_act - x_goal);
  return u_output;
}

ADAPTIVE_DDP_TEMPLATE
void AdaptiveDDP::computeFeedback(const Eigen::Ref<const state_array>& init_state,
                                  const Eigen::Ref<const state_trajectory>& goal_traj,
                                  const Eigen::Ref<const control_trajectory>& control_traj)
{
  baseline_ddp_->computeFeedback(init_state, goal_traj, control_traj);
  auto* s = this->getFeedbackStatePointer();
  std::memcpy(s->fb_gain_traj_, baseline_ddp_->getFeedbackStatePointer()->fb_gain_traj_,
              sizeof(float) * INTERNAL_STATE_T::GAIN_SIZE);

  const auto& result = baseline_ddp_->result_;
  std::vector<typename Reference::feedback_matrix> K;
  typename Reference::state_traj x_ref = goal_traj;
  typename Reference::control_traj u_ref = control_traj;
  const int solved = static_cast<int>(result.feedback_gain.size());
  for (int t = 0; t < this->num_timesteps_; t++)
  {
    K.push_back(t < solved ? result.feedback_gain[t] : feedback_gain_matrix::Zero());
    // Validity is measured against the trajectory the DDP gains were linearized about.
    if (t < result.state_trajectory.cols())
    {
      x_ref.col(t) = result.state_trajectory.col(t);
      u_ref.col(t) = result.control_trajectory.col(t);
    }
    for (int i = 0; i < DYN_T::STATE_DIM; i++)
    {
      s->x_ref_traj_[DYN_T::STATE_DIM * t + i] = x_ref(i, t);
    }
  }
  reference_->setBaselineCache(K, x_ref, u_ref);
  reference_->setRefreshReference(goal_traj, control_traj);
}

ADAPTIVE_DDP_TEMPLATE
mppi::adaptive::RefreshOutcome AdaptiveDDP::refreshPolicy(int slot, int t, const state_array& x, int base_slot)
{
  typename Reference::Cache cache = reference_->getBaselineCache();
  auto outcome = reference_->refresh(cache, t, x);
  if (outcome != mppi::adaptive::RefreshOutcome::REFRESHED)
  {
    return outcome;
  }
  auto* s = this->getFeedbackStatePointer();
  const int SC = DYN_T::STATE_DIM * DYN_T::CONTROL_DIM;
  float* gains = &s->policy_gains_[slot * INTERNAL_STATE_T::GAIN_SIZE];
  float* x_ref = &s->policy_x_ref_[slot * INTERNAL_STATE_T::REF_SIZE];
  const float* base_gains =
      base_slot < 0 ? s->fb_gain_traj_ : &s->policy_gains_[base_slot * INTERNAL_STATE_T::GAIN_SIZE];
  const float* base_x_ref =
      base_slot < 0 ? s->x_ref_traj_ : &s->policy_x_ref_[base_slot * INTERNAL_STATE_T::REF_SIZE];
  std::memcpy(gains, base_gains, sizeof(float) * INTERNAL_STATE_T::GAIN_SIZE);
  std::memcpy(x_ref, base_x_ref, sizeof(float) * INTERNAL_STATE_T::REF_SIZE);
  for (int k = 0; k < cache.size(); k++)
  {
    const int abs_t = cache.start_t + k;
    std::memcpy(&gains[abs_t * SC], cache.K[k].data(), sizeof(float) * SC);
    std::memcpy(&x_ref[abs_t * DYN_T::STATE_DIM], cache.x_ref[k].data(), sizeof(float) * DYN_T::STATE_DIM);
  }
  this->gpu_controller_->copyPolicyToDevice(slot, false);
  return outcome;
}

ADAPTIVE_DDP_TEMPLATE
typename AdaptiveDDP::feedback_gain_matrix AdaptiveDDP::getPolicyGain(int slot, int t)
{
  auto* s = this->getFeedbackStatePointer();
  const int SC = DYN_T::STATE_DIM * DYN_T::CONTROL_DIM;
  float* base = slot < 0 ? s->fb_gain_traj_ : &s->policy_gains_[slot * INTERNAL_STATE_T::GAIN_SIZE];
  return Eigen::Map<feedback_gain_matrix>(&base[t * SC]);
}

#undef AdaptiveDDP
#undef ADAPTIVE_DDP_TEMPLATE
