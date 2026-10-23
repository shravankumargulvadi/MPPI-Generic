// Standalone source-level mechanism probe; include the exact benchmark types/setup.
#define main heading_benchmark_main
#include <examples/adaptive_rmppi_heading_benchmark.cu>
#undef main

int main()
{
  DYN model;
  setupModel(model);
  COST cost;
  setupCost(cost);
  SAMPLING::SAMPLING_PARAMS_T sp;
  sp.std_dev[0] = STD_DEV_VEL;
  sp.std_dev[1] = STD_DEV_YAW_RATE;
  SAMPLING sampler(sp);
  ADAPTIVE_FB fb(&model, DT);
  setFbParams(fb);
  ADAPTIVE c(&model, &cost, &fb, &sampler, DT, 1, LAMBDA, ALPHA, VALUE_FUNCTION_THRESHOLD);
  configure(c, 42);
  ADAPTIVE_FB::Config cfg;
  cfg.enabled = true;
  cfg.distance_radius = 0.5f;
  cfg.max_refreshes_per_iteration = 0;
  cfg.state_weights[0] = 0;
  cfg.angle_state_mask[2] = 1;
  c.setAdaptiveConfig(cfg);
  DYN::state_array x;
  x << 0, 0, 0;
  c.updateImportanceSamplingControl(x, 1);
  c.computeControl(x, 1);
  x(2) = 2.5f;
  c.updateImportanceSamplingControl(x, 1);
  c.computeControl(x, 1);
  const auto target = c.getTargetStateSeq().col(0).eval();
  const auto baseline_gain = fb.getBaselineDDP().result_.feedback_gain[0];
  const auto baseline_action = fb.getBaselineDDP().result_.control_trajectory.col(0).eval();
  const auto baseline_start = fb.getBaselineDDP().result_.state_trajectory.col(0).eval();
  ADAPTIVE_FB::Reference::Cache recovery;
  const auto outcome = fb.solveRecovery(0, x, recovery);
  if (outcome != mppi::adaptive::RefreshOutcome::REFRESHED) return 1;
  const auto feedback = c.getFeedbackControl(x, target, 0);
  const auto applied = (c.getControlSeq().col(0) + feedback).eval();
  std::cout << std::setprecision(10)
            << "baseline_initial_state_error=" << (baseline_start-x).norm() << "\n"
            << "recovery_vs_baseline_gain_error=" << (recovery.K[0]-baseline_gain).norm() << "\n"
            << "recovery_vs_baseline_ddp_action_error=" << (recovery.u_ref[0]-baseline_action).norm() << "\n"
            << "applied_vs_recovery_action_error=" << (applied-recovery.u_ref[0]).norm() << "\n";
  c.getFeedbackControl(x, target, 0);
  std::cout << "plant_refreshes_after_two_calls=" << c.getLastIterationStats().plant_refreshes << "\n";
  const bool ok = (baseline_start-x).norm()<1e-6f && (recovery.K[0]-baseline_gain).norm()<1e-5f &&
      (recovery.u_ref[0]-baseline_action).norm()<1e-5f && (applied-recovery.u_ref[0]).norm()<1e-5f &&
      c.getLastIterationStats().plant_refreshes == 2;
  cfg.plant_feedback = false;
  c.setAdaptiveConfig(cfg);
  const auto disabled = c.getFeedbackControl(x, target, 0);
  std::cout << "plant_disabled_feedback_error=" << (disabled-baseline_gain*(x-target)).norm() << "\n";
  return ok && (disabled-baseline_gain*(x-target)).norm()<1e-5f ? 0 : 1;
}
