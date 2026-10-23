/**
 * RMPPI with event-triggered, per-rollout iLQR gain refresh (staged GPU design).
 *
 * With the adaptive config disabled this controller is RobustMPPIController: launchRolloutKernels() defers to the
 * parent and the feedback device code performs the stock DDP tracking law.
 *
 * With it enabled each optimization iteration runs:
 *   pass A: rollouts with the current per-rollout policies; the feedback controller records the first timestep and
 *           state at which each actual rollout leaves the V0 validity radius,
 *   refresh: the host re-solves remaining-horizon DDP from each recorded state (bounded per rollout and per
 *           iteration) and writes the result into a policy-bank slot owned by that rollout,
 *   pass B: the sampled noise is restored and all rollouts are re-run from t = 0, so every cost, likelihood-ratio and
 *           feedback-cost term comes from the trajectory that was actually simulated.
 * Passes repeat until no rollout changes policy or max_refreshes_per_rollout is reached.
 * Adaptive mode requires the combined rollout kernel (policy lookup is keyed by sample); it is selected automatically and
 * a std::runtime_error is thrown if it does not fit in shared memory.
 * See doc/adaptive_rmppi_implementation_note.md for limitations.
 */
#pragma once

#include <mppi/controllers/R-MPPI/robust_mppi_controller.cuh>
#include <mppi/feedback_controllers/adaptive/adaptive_ddp.cuh>

#include <vector>

namespace mppi
{
namespace adaptive
{
struct AdaptiveRolloutStats
{
  int iterations = 0;
  int passes = 0;
  int violations = 0;
  int refreshes = 0;
  int solver_failures = 0;
  int budget_overflows = 0;
  int horizon_too_short = 0;
  // Plant-side recovery solves (getFeedbackControl); counted separately from rollout refreshes.
  int plant_refreshes = 0;
  int plant_refresh_failures = 0;
  std::vector<RefreshEvent> events;

  void accumulate(const AdaptiveRolloutStats& o)
  {
    iterations += o.iterations;
    passes += o.passes;
    violations += o.violations;
    refreshes += o.refreshes;
    solver_failures += o.solver_failures;
    budget_overflows += o.budget_overflows;
    horizon_too_short += o.horizon_too_short;
    plant_refreshes += o.plant_refreshes;
    plant_refresh_failures += o.plant_refresh_failures;
  }
};
}  // namespace adaptive
}  // namespace mppi

template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS = 2560,
          class SAMPLING_T = ::mppi::sampling_distributions::GaussianDistribution<typename DYN_T::DYN_PARAMS_T>,
          class PARAMS_T = RobustMPPIParams<DYN_T::STATE_DIM, DYN_T::CONTROL_DIM, MAX_TIMESTEPS>>
class AdaptiveRobustMPPIController
  : public RobustMPPIController<DYN_T, COST_T, FB_T, MAX_TIMESTEPS, NUM_ROLLOUTS, SAMPLING_T, PARAMS_T>
{
public:
  typedef RobustMPPIController<DYN_T, COST_T, FB_T, MAX_TIMESTEPS, NUM_ROLLOUTS, SAMPLING_T, PARAMS_T> PARENT_CLASS;
  using FEEDBACK_GPU = typename PARENT_CLASS::FEEDBACK_GPU;
  using Config = typename FB_T::Config;
  using Stats = mppi::adaptive::AdaptiveRolloutStats;
  using state_array = typename PARENT_CLASS::state_array;
  using control_array = typename PARENT_CLASS::control_array;

  using PARENT_CLASS::PARENT_CLASS;

  ~AdaptiveRobustMPPIController();

  std::string getControllerName()
  {
    return "Adaptive Robust MPPI";
  };

  void setAdaptiveConfig(const Config& config)
  {
    this->fb_controller_->setAdaptiveConfig(config);
  }
  const Config& getAdaptiveConfig() const
  {
    return this->fb_controller_->getAdaptiveConfig();
  }

  // Statistics of the most recent optimization iteration, and accumulated since the last reset.
  const Stats& getLastIterationStats() const
  {
    return last_stats_;
  }
  const Stats& getAccumulatedStats() const
  {
    return total_stats_;
  }
  void resetAccumulatedStats()
  {
    total_stats_ = Stats();
  }

  /**
   * Plant-side tracking term. Stock behaviour (baseline K_t (x - x*_t)) unless adaptive mode is enabled, plant
   * feedback is on and the plant state is outside the trigger radius; then a remaining-horizon recovery policy is
   * solved from the plant state and its first action replaces the baseline correction.
   */
  control_array getFeedbackControl(const Eigen::Ref<const state_array>& state,
                                   const Eigen::Ref<const state_array>& goal_state, int t) override;

protected:
  void launchRolloutKernels() override;
  // Throws if the combined kernel does not fit in shared memory; otherwise selects it.
  void ensureCombinedKernel();
  void launchSingleKernel();
  void copyPolicyAssignmentToDevice(const std::vector<int>& policy_id, const std::vector<int>& policy_start);

  size_t shared_mem_per_block_ = 0;
  float* saved_samples_d_ = nullptr;
  size_t saved_samples_size_ = 0;
  Stats last_stats_;
  Stats total_stats_;
};

#if __CUDACC__
#include "adaptive_robust_mppi_controller.cu"
#endif
