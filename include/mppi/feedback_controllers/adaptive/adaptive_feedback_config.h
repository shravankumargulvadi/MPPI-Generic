/**
 * Configuration for event-triggered, trajectory-local iLQR gain refresh inside RMPPI rollouts.
 * See doc/adaptive_rmppi_implementation_note.md for the algorithm and its limitations.
 */
#pragma once

#include <cmath>

namespace mppi
{
namespace adaptive
{
enum class TriggerType : int
{
  // V0: refresh when sqrt((x - x~)^T W (x - x~)) > distance_radius
  DISTANCE = 0,
  // V1: refresh when ||F(x,u) - [F(x~,u~) + A(x - x~) + B(u - u~)]||_W > residual_threshold
  //     or the V0 distance exceeds hard_distance_radius
  MODEL_RESIDUAL,
};

enum class RefreshFallback : int
{
  // Keep using the previous (stale) gain sequence for the remainder of the rollout.
  KEEP_PREVIOUS_GAINS = 0,
};

template <int S_DIM>
struct AdaptiveFeedbackConfig
{
  // When false, the controller must behave exactly like stock RMPPI.
  bool enabled = false;
  TriggerType trigger = TriggerType::DISTANCE;
  // Diagonal of W used for both distance and residual norms. Units: 1 / (state unit)^2.
  float state_weights[S_DIM];
  // Nonzero entries mark angle states whose differences are wrapped to [-pi, pi).
  int angle_state_mask[S_DIM];
  // V0 trigger radius (dimensionless, in W-norm). A tuning knob, not a stability certificate.
  float distance_radius = 1.0f;
  // V1 trigger threshold on the W-norm of the one-step linearization residual.
  float residual_threshold = 0.1f;
  // V1 also refreshes when the distance exceeds this radius.
  float hard_distance_radius = 10.0f;
  // Per-rollout cap on gain refreshes within one optimization iteration.
  int max_refreshes_per_rollout = 2;
  // Per-iteration cap on gain refreshes across all rollouts. Overflows are counted, never hidden.
  int max_refreshes_per_iteration = 64;
  RefreshFallback fallback = RefreshFallback::KEEP_PREVIOUS_GAINS;
  // Also re-solve the plant-side tracking policy when the real state leaves the radius around the nominal state
  // (at most one CPU solve per control cycle). Only used when `enabled`.
  bool plant_feedback = true;

  AdaptiveFeedbackConfig()
  {
    for (int i = 0; i < S_DIM; i++)
    {
      state_weights[i] = 1.0f;
      angle_state_mask[i] = 0;
    }
  }
};

/**
 * Weighted state difference a - b with angle wrapping, written into diff. Returns the W-norm.
 */
template <int S_DIM>
inline __host__ __device__ float weightedStateDistance(const float* a, const float* b, const float* weights,
                                                       const int* angle_mask, float* diff = nullptr)
{
  float sq = 0.0f;
  for (int i = 0; i < S_DIM; i++)
  {
    float d = a[i] - b[i];
    if (angle_mask[i])
    {
      d = atan2f(sinf(d), cosf(d));
    }
    if (diff)
    {
      diff[i] = d;
    }
    sq += weights[i] * d * d;
  }
  return sqrtf(sq);
}
}  // namespace adaptive
}  // namespace mppi
