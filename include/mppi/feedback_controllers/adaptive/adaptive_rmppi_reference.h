/**
 * CPU reference ("oracle") for event-triggered gain refresh in RMPPI rollouts.
 *
 * One call to rollout() simulates one sample the way rolloutRMPPIKernel does: the nominal state x* is
 * propagated without feedback and the actual state x is propagated with
 *   u = v + K~_t (x - x*_t)
 * where K~ is the gain of the rollout's own cached policy. The cache also stores the recovery trajectory
 * (x~, u~) that K~ was linearized about. When the validity trigger fires, the remaining-horizon iLQR problem
 * is re-solved from the current actual state and the cache is replaced for this rollout only.
 * The tracking target stays x*; only the gain changes.
 */
#pragma once

#include <mppi/feedback_controllers/DDP/ddp.cuh>
#include <mppi/feedback_controllers/adaptive/adaptive_feedback_config.h>

#include <Eigen/Dense>
#include <functional>
#include <vector>

namespace mppi
{
namespace adaptive
{
enum class RefreshOutcome : int
{
  REFRESHED = 0,
  BUDGET_EXCEEDED,     // trigger fired but the per-rollout or per-iteration budget was exhausted
  SOLVER_FAILED,       // iLQR returned non-finite gains/states; previous gains kept
  HORIZON_TOO_SHORT,   // fewer than 2 timesteps remain; previous gains kept
};

struct RefreshEvent
{
  int sample = -1;
  int t = -1;
  float distance = 0.0f;
  float residual = 0.0f;
  RefreshOutcome outcome = RefreshOutcome::REFRESHED;
};

template <class DYN_T>
struct GainCache
{
  using state_array = typename DYN_T::state_array;
  using control_array = typename DYN_T::control_array;
  using feedback_matrix = Eigen::Matrix<float, DYN_T::CONTROL_DIM, DYN_T::STATE_DIM>;
  using state_matrix = Eigen::Matrix<float, DYN_T::STATE_DIM, DYN_T::STATE_DIM>;
  using control_matrix = Eigen::Matrix<float, DYN_T::STATE_DIM, DYN_T::CONTROL_DIM>;

  // Absolute horizon index of the first entry.
  int start_t = 0;
  std::vector<state_array> x_ref;
  std::vector<control_array> u_ref;
  std::vector<feedback_matrix> K;
  // Discrete-time Jacobians at (x_ref, u_ref); only filled for the residual trigger.
  std::vector<state_matrix> A_d;
  std::vector<control_matrix> B_d;

  int size() const
  {
    return static_cast<int>(K.size());
  }
  bool covers(int t) const
  {
    return t >= start_t && t - start_t < size();
  }
};

template <class DYN_T>
class AdaptiveRMPPIReference
{
public:
  static const int S_DIM = DYN_T::STATE_DIM;
  static const int C_DIM = DYN_T::CONTROL_DIM;
  using state_array = typename DYN_T::state_array;
  using control_array = typename DYN_T::control_array;
  using output_array = typename DYN_T::output_array;
  using Cache = GainCache<DYN_T>;
  using feedback_matrix = typename Cache::feedback_matrix;
  using state_traj = Eigen::Matrix<float, S_DIM, Eigen::Dynamic>;
  using control_traj = Eigen::Matrix<float, C_DIM, Eigen::Dynamic>;
  using Config = AdaptiveFeedbackConfig<S_DIM>;
  using DDPWrapper = ModelWrapperDDP<DYN_T>;

  // Solves the tracking problem for the suffix starting at absolute index t0 from x0. The reference
  // trajectories passed in are already the suffix (column 0 == absolute index t0). Returns false on failure.
  using GainSolver = std::function<bool(const state_array& x0, int t0, const state_traj& x_goal,
                                        const control_traj& u_goal, Cache& out)>;

  struct RolloutResult
  {
    state_traj x_nom;       // S x (T + 1)
    state_traj x_act;       // S x (T + 1)
    control_traj u_applied; // C x T, actual-rollout control after feedback and constraints
    control_traj u_fb;      // C x T, feedback term before constraints (what the FB cost uses)
    std::vector<RefreshEvent> events;
    int num_refreshes = 0;
  };

  AdaptiveRMPPIReference(DYN_T* model, float dt, int num_timesteps, const Config& config,
                         const DDPParams<DYN_T>& ddp_params)
    : model_(model), dt_(dt), num_timesteps_(num_timesteps), config_(config), ddp_params_(ddp_params)
  {
    ddp_model_ = std::make_shared<DDPWrapper>(model_);
    solver_ = [this](const state_array& x0, int t0, const state_traj& x_goal, const control_traj& u_goal,
                     Cache& out) { return solveDDP(x0, t0, x_goal, u_goal, out); };
  }

  void setSolver(GainSolver solver)
  {
    solver_ = solver;
  }
  void setConfig(const Config& config)
  {
    config_ = config;
  }
  const Config& getConfig() const
  {
    return config_;
  }

  /**
   * Baseline policy shared by all rollouts at the start of an iteration: the gains RMPPI computed against its
   * nominal trajectory, plus the trajectory those gains were linearized about.
   */
  void setBaselineCache(const std::vector<feedback_matrix>& K, const state_traj& x_ref, const control_traj& u_ref)
  {
    baseline_ = Cache();
    baseline_.start_t = 0;
    for (int t = 0; t < num_timesteps_; t++)
    {
      baseline_.K.push_back(K[t]);
      baseline_.x_ref.push_back(x_ref.col(t));
      baseline_.u_ref.push_back(u_ref.col(t));
    }
    fillJacobians(baseline_);
  }

  // Reference the refresh solver tracks (RMPPI's noise-free nominal trajectory and controls).
  void setRefreshReference(const state_traj& x_goal, const control_traj& u_goal)
  {
    refresh_x_goal_ = x_goal;
    refresh_u_goal_ = u_goal;
  }

  void resetIterationBudget()
  {
    iteration_refreshes_ = 0;
  }
  int getIterationRefreshes() const
  {
    return iteration_refreshes_;
  }

  /**
   * Trigger metrics for actual state x at absolute timestep t under cache c. prev_* describe the previous
   * actual step (used by the residual trigger); pass t == c.start_t to skip the residual.
   */
  void triggerMetrics(const Cache& c, int t, const state_array& x, const state_array& x_prev,
                      const control_array& u_prev, float& distance, float& residual) const
  {
    int k = t - c.start_t;
    distance = weightedStateDistance<S_DIM>(x.data(), c.x_ref[k].data(), config_.state_weights,
                                            config_.angle_state_mask);
    residual = 0.0f;
    if (config_.trigger == TriggerType::MODEL_RESIDUAL && k >= 1)
    {
      state_array dx_prev;
      weightedStateDistance<S_DIM>(x_prev.data(), c.x_ref[k - 1].data(), config_.state_weights,
                                   config_.angle_state_mask, dx_prev.data());
      state_array predicted = c.x_ref[k] + c.A_d[k - 1] * dx_prev + c.B_d[k - 1] * (u_prev - c.u_ref[k - 1]);
      residual = weightedStateDistance<S_DIM>(x.data(), predicted.data(), config_.state_weights,
                                              config_.angle_state_mask);
    }
  }

  bool triggerFires(float distance, float residual) const
  {
    if (config_.trigger == TriggerType::DISTANCE)
    {
      return distance > config_.distance_radius;
    }
    return residual > config_.residual_threshold || distance > config_.hard_distance_radius;
  }

  /**
   * Paired nominal/actual rollout for one sample. v_nom / v_act are the sampled controls (C x T) for the
   * nominal and actual distributions, mirroring the RMPPI kernel.
   */
  RolloutResult rollout(int sample, const state_array& x_nom0, const state_array& x_act0, const control_traj& v_nom,
                        const control_traj& v_act)
  {
    RolloutResult r;
    const int T = num_timesteps_;
    r.x_nom = state_traj::Zero(S_DIM, T + 1);
    r.x_act = state_traj::Zero(S_DIM, T + 1);
    r.u_applied = control_traj::Zero(C_DIM, T);
    r.u_fb = control_traj::Zero(C_DIM, T);
    r.x_nom.col(0) = x_nom0;
    r.x_act.col(0) = x_act0;

    Cache cache = baseline_;
    bool budget_logged = false;
    state_array x_nom = x_nom0, x_act = x_act0, next, der;
    output_array out;
    control_array u_prev = control_array::Zero();

    for (int t = 0; t < T; t++)
    {
      if (config_.enabled)
      {
        float distance, residual;
        triggerMetrics(cache, t, x_act, r.x_act.col(t > 0 ? t - 1 : 0), u_prev, distance, residual);
        if (triggerFires(distance, residual))
        {
          RefreshEvent e;
          e.sample = sample;
          e.t = t;
          e.distance = distance;
          e.residual = residual;
          if (r.num_refreshes >= config_.max_refreshes_per_rollout ||
              iteration_refreshes_ >= config_.max_refreshes_per_iteration)
          {
            e.outcome = RefreshOutcome::BUDGET_EXCEEDED;
            if (!budget_logged)
            {
              r.events.push_back(e);
              budget_logged = true;
            }
          }
          else
          {
            e.outcome = refresh(cache, t, x_act);
            iteration_refreshes_ += (e.outcome == RefreshOutcome::HORIZON_TOO_SHORT) ? 0 : 1;
            r.num_refreshes += (e.outcome == RefreshOutcome::HORIZON_TOO_SHORT) ? 0 : 1;
            r.events.push_back(e);
          }
        }
      }

      control_array u_nom = v_nom.col(t);
      model_->enforceConstraints(x_nom, u_nom);
      model_->step(x_nom, next, der, u_nom, out, t, dt_);
      x_nom = next;

      control_array fb = cache.K[t - cache.start_t] * (x_act - r.x_nom.col(t));
      control_array u = v_act.col(t) + fb;
      model_->enforceConstraints(x_act, u);
      r.u_fb.col(t) = fb;
      r.u_applied.col(t) = u;
      model_->step(x_act, next, der, u, out, t, dt_);
      x_act = next;
      u_prev = u;

      r.x_nom.col(t + 1) = x_nom;
      r.x_act.col(t + 1) = x_act;
    }
    return r;
  }

  const Cache& getBaselineCache() const
  {
    return baseline_;
  }

  /**
   * Replace cache with a policy re-solved from x_act at absolute index t. On failure the cache is unchanged
   * (fallback: keep previous gains).
   */
  RefreshOutcome refresh(Cache& cache, int t, const state_array& x_act)
  {
    const int H = num_timesteps_ - t;
    if (H < 2)
    {
      return RefreshOutcome::HORIZON_TOO_SHORT;
    }
    Cache fresh;
    bool ok = solver_(x_act, t, refresh_x_goal_.block(0, t, S_DIM, H), refresh_u_goal_.block(0, t, C_DIM, H), fresh);
    ok = ok && fresh.size() == H && fresh.start_t == t && isFinite(fresh);
    if (!ok)
    {
      return RefreshOutcome::SOLVER_FAILED;
    }
    fillJacobians(fresh);
    cache = fresh;
    return RefreshOutcome::REFRESHED;
  }

  // Default solver: the same DDP tracking problem DDPFeedback solves, restricted to the remaining horizon.
  bool solveDDP(const state_array& x0, int t0, const state_traj& x_goal, const control_traj& u_goal, Cache& out)
  {
    const int H = x_goal.cols();
    ::util::DefaultLogger logger;
    DDP<DDPWrapper> solver(dt_, H, ddp_params_.num_iterations, &logger, false);
    TrackingCostDDP<DDPWrapper> run_cost(ddp_params_.Q, ddp_params_.R, H);
    TrackingTerminalCost<DDPWrapper> term_cost(ddp_params_.Q_f);
    Eigen::MatrixXf xg = x_goal, ug = u_goal;
    run_cost.setTargets(xg.data(), ug.data(), H);
    term_cost.xf = run_cost.traj_target_x_.col(H - 1);
    control_array u_min, u_max;
    for (int i = 0; i < C_DIM; i++)
    {
      u_min(i) = model_->control_rngs_[i].x;
      u_max(i) = model_->control_rngs_[i].y;
    }
    auto result = solver.run(x0, ug, *ddp_model_, run_cost, term_cost, u_min, u_max);
    out = Cache();
    out.start_t = t0;
    for (int k = 0; k < H; k++)
    {
      out.K.push_back(result.feedback_gain[k]);
      out.x_ref.push_back(result.state_trajectory.col(k));
      out.u_ref.push_back(result.control_trajectory.col(k));
    }
    return true;
  }

private:
  static bool isFinite(const Cache& c)
  {
    for (int k = 0; k < c.size(); k++)
    {
      if (!c.K[k].allFinite() || !c.x_ref[k].allFinite() || !c.u_ref[k].allFinite())
      {
        return false;
      }
    }
    return true;
  }

  void fillJacobians(Cache& c)
  {
    c.A_d.clear();
    c.B_d.clear();
    if (config_.trigger != TriggerType::MODEL_RESIDUAL)
    {
      return;
    }
    for (int k = 0; k < c.size(); k++)
    {
      auto J = ddp_model_->df(c.x_ref[k], c.u_ref[k]);
      c.A_d.push_back(Cache::state_matrix::Identity() + J.template leftCols<S_DIM>() * dt_);
      c.B_d.push_back(J.template rightCols<C_DIM>() * dt_);
    }
  }

  DYN_T* model_;
  float dt_;
  int num_timesteps_;
  Config config_;
  DDPParams<DYN_T> ddp_params_;
  std::shared_ptr<DDPWrapper> ddp_model_;
  GainSolver solver_;
  Cache baseline_;
  state_traj refresh_x_goal_;
  control_traj refresh_u_goal_;
  int iteration_refreshes_ = 0;
};
}  // namespace adaptive
}  // namespace mppi
