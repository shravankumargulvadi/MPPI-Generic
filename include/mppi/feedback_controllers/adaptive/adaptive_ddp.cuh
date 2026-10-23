/**
 * DDP feedback with a bank of per-rollout refreshed policies for adaptive RMPPI.
 *
 * Baseline gains are the same DDP gains DDPFeedback computes; with adaptation disabled the device k() performs
 * exactly the DDP tracking law. With adaptation enabled each rollout reads its own policy (bank slot + start time)
 * and, in recording passes, reports the first timestep at which its actual state leaves the V0 validity radius of the
 * trajectory its current gains were linearized about. No thread ever writes gains; the host refreshes the bank
 * between passes (see AdaptiveRobustMPPIController).
 */
#pragma once

#include <mppi/feedback_controllers/DDP/ddp.cuh>
#include <mppi/feedback_controllers/adaptive/adaptive_feedback_config.h>
#include <mppi/feedback_controllers/adaptive/adaptive_rmppi_reference.h>

template <class DYN_T, int N_TIMESTEPS, int MAX_POLICIES>
struct AdaptiveDDPFeedbackState : GPUState
{
  static const int S_DIM = DYN_T::STATE_DIM;
  static const int C_DIM = DYN_T::CONTROL_DIM;
  static const int NUM_TIMESTEPS = N_TIMESTEPS;
  static const int NUM_POLICIES = MAX_POLICIES;
  static const int GAIN_SIZE = S_DIM * C_DIM * N_TIMESTEPS;
  static const int REF_SIZE = S_DIM * N_TIMESTEPS;
  // Same layout as DDPFeedbackState::fb_gain_traj_ (column-major C x S per timestep).
  float fb_gain_traj_[GAIN_SIZE] MPPI_ALIGN(16) = { 0.0f };
  // Trajectory the baseline gains were linearized about.
  float x_ref_traj_[REF_SIZE] = { 0.0f };
  // Refreshed policies, indexed by absolute timestep; entries before the policy's start time are unused.
  float policy_gains_[MAX_POLICIES * GAIN_SIZE] MPPI_ALIGN(16) = { 0.0f };
  float policy_x_ref_[MAX_POLICIES * REF_SIZE] = { 0.0f };
  float state_weights_[S_DIM];
  int angle_mask_[S_DIM];
  float distance_radius_ = 1.0f;
  int num_rollouts_ = 0;
  int adaptive_enabled_ = 0;
  int record_violations_ = 0;
  // Per-rollout device buffers of length num_rollouts.
  int* policy_id_d_ = nullptr;        // -1: baseline gains
  int* policy_start_t_d_ = nullptr;   // first timestep the policy applies to
  int* violation_t_d_ = nullptr;      // -1: no violation recorded this pass
  float* violation_x_d_ = nullptr;    // actual state at violation_t, S_DIM per rollout
};

template <class GPU_FB_T, class DYN_T, int NUM_TIMESTEPS, int MAX_POLICIES>
class DeviceAdaptiveDDPImpl
  : public GPUFeedbackController<GPU_FB_T, DYN_T, AdaptiveDDPFeedbackState<DYN_T, NUM_TIMESTEPS, MAX_POLICIES>>
{
public:
  using STATE_T = AdaptiveDDPFeedbackState<DYN_T, NUM_TIMESTEPS, MAX_POLICIES>;
  using PARENT_CLASS = GPUFeedbackController<GPU_FB_T, DYN_T, STATE_T>;

  DeviceAdaptiveDDPImpl(cudaStream_t stream = 0) : PARENT_CLASS(stream)
  {
  }

  void allocateCUDAMemory()
  {
    allocateRolloutBuffers();
  }
  void deallocateCUDAMemory();

  // Must be called before rollout buffers are used; reallocates if already on the GPU.
  void setNumRollouts(int num_rollouts);
  int getNumRollouts() const
  {
    return num_rollouts_;
  }

  // Small, targeted host->device copies so a refresh does not resend the whole policy bank.
  void copyPolicyToDevice(int slot, bool synchronize = false);
  void copyFlagsToDevice(bool synchronize = false);

  __device__ void k(const float* __restrict__ x_act, const float* __restrict__ x_goal, const int t,
                    float* __restrict__ theta, float* __restrict__ control_output);

protected:
  void allocateRolloutBuffers();
  int num_rollouts_ = 0;
};

template <class DYN_T, int NUM_TIMESTEPS, int MAX_POLICIES>
class DeviceAdaptiveDDP
  : public DeviceAdaptiveDDPImpl<DeviceAdaptiveDDP<DYN_T, NUM_TIMESTEPS, MAX_POLICIES>, DYN_T, NUM_TIMESTEPS,
                                 MAX_POLICIES>
{
public:
  DeviceAdaptiveDDP(cudaStream_t stream = 0)
    : DeviceAdaptiveDDPImpl<DeviceAdaptiveDDP<DYN_T, NUM_TIMESTEPS, MAX_POLICIES>, DYN_T, NUM_TIMESTEPS,
                            MAX_POLICIES>(stream)
  {
  }
};

template <class DYN_T, int NUM_TIMESTEPS, int MAX_POLICIES = 64>
class AdaptiveDDPFeedback
  : public FeedbackController<DeviceAdaptiveDDP<DYN_T, NUM_TIMESTEPS, MAX_POLICIES>, DDPParams<DYN_T>, NUM_TIMESTEPS>
{
public:
  typedef FeedbackController<DeviceAdaptiveDDP<DYN_T, NUM_TIMESTEPS, MAX_POLICIES>, DDPParams<DYN_T>, NUM_TIMESTEPS>
      PARENT_CLASS;
  using control_array = typename PARENT_CLASS::control_array;
  using state_array = typename PARENT_CLASS::state_array;
  using state_trajectory = typename PARENT_CLASS::state_trajectory;
  using control_trajectory = typename PARENT_CLASS::control_trajectory;
  using INTERNAL_STATE_T = typename PARENT_CLASS::TEMPLATED_FEEDBACK_STATE;
  using feedback_gain_matrix = typename DYN_T::feedback_matrix;
  using Config = mppi::adaptive::AdaptiveFeedbackConfig<DYN_T::STATE_DIM>;
  using Reference = mppi::adaptive::AdaptiveRMPPIReference<DYN_T>;
  static const int MAX_POLICY_SLOTS = MAX_POLICIES;

  AdaptiveDDPFeedback(DYN_T* model, float dt, int num_timesteps = NUM_TIMESTEPS, cudaStream_t stream = 0);

  void initTrackingController();
  void setParams(const DDPParams<DYN_T>& params);
  void setAdaptiveConfig(const Config& config);
  const Config& getAdaptiveConfig() const
  {
    return config_;
  }

  control_array k_(const Eigen::Ref<const state_array>& x_act, const Eigen::Ref<const state_array>& x_goal, int t,
                   INTERNAL_STATE_T& fb_state);

  // Baseline gains: identical problem and solver to DDPFeedback::computeFeedback.
  void computeFeedback(const Eigen::Ref<const state_array>& init_state,
                       const Eigen::Ref<const state_trajectory>& goal_traj,
                       const Eigen::Ref<const control_trajectory>& control_traj);

  /**
   * Remaining-horizon re-solve from actual state x at absolute timestep t, written into bank slot `slot`.
   * Returns the outcome; the slot is only written on REFRESHED.
   */
  // Writes a refreshed policy into `slot`: gains before t are copied from `base_slot` (-1 = baseline), so earlier
  // refreshes of the same rollout are preserved.
  mppi::adaptive::RefreshOutcome refreshPolicy(int slot, int t, const state_array& x, int base_slot = -1);

  // Host copy of a stored policy gain (baseline when slot < 0) for tests and logging.
  feedback_gain_matrix getPolicyGain(int slot, int t);
  // Remaining-horizon recovery solve from x at timestep t against the current nominal reference. Does not touch the
  // policy bank; used for the plant-side tracking term.
  mppi::adaptive::RefreshOutcome solveRecovery(int t, const state_array& x, typename Reference::Cache& cache);

  Reference& getReference()
  {
    return *reference_;
  }
  DDPFeedback<DYN_T, NUM_TIMESTEPS>& getBaselineDDP()
  {
    return *baseline_ddp_;
  }

protected:
  DYN_T* model_;
  Config config_;
  std::shared_ptr<DDPFeedback<DYN_T, NUM_TIMESTEPS>> baseline_ddp_;
  std::shared_ptr<Reference> reference_;
};

#if __CUDACC__
#include "adaptive_ddp.cu"
#endif
