# Cart-pole trigger calibration and parameter study

Starting implementation: `26d7221` (controller code from `7c54a0f`). Only the benchmark harness and offline analysis are changed for this study. Paired modes share seeds, plant, horizon, cost, force limits, sampling settings, DDP iteration count and recovery criteria. The primary recovery criterion remains the original weighted full-state error <0.2 by 2 seconds after the kick, maintained through the remaining episode; pole-only recovery is secondary.

## Calibration and selection

1. Initial radius sweep on seeds 0-2: radii 0.5, 1, 2, 4, 8, 16, 32, and 1e6; nominal, matched 1 rad/s, mismatched 1 rad/s. This found radius 4 suppressed nominal demand to about 0.074 refresh/cycle, while radius 8 suppressed all demand.
2. Refine radii 3.5, 4, 4.5, 5, and 1e6 on seeds 10-14. Choose the smallest radius with nominal mean <=0.1 refresh/cycle and no nominal budget hits; inspect whether disturbances still trigger it. Selection uses demand, not a favorable adaptive outcome. Distance alone is not treated as a certificate that a gain is invalid.
3. Explore common controller/plant settings on seeds 20-22, keeping the selected radius: default; force 20 N; cart weight 200; noise stddev 2.5 and 7.5; lambda 1; three DDP iterations; refresh budget 8; horizons 2 and 3 seconds. Use kicks 0.5, 1, and 1.5 rad/s, matched and mismatched plants, plus nominal. These are exploratory data. A longer horizon may expose more nominal trigger demand; report that rather than hiding it.
4. If necessary, perform a follow-up sweep on development seeds below 1000 to resolve a promising or failing regime. Record the reason and settings before each follow-up. At most two parameter configurations proceed to final evaluation: the baseline configuration with the tuned trigger, and one shared configuration selected for useful stock recovery (plus any clear adaptive benefit seen in development data). No held-out data influences selection.

## Final evaluation

Freeze the configurations and trigger radii before running seeds 5000-5029, which have not been used in earlier runs. Include nominal and matched/mismatched kicks around the development recovery transition. Compare paired primary and pole-only success, recovery time, executed cost, peak position/angle, control saturation, mean trial p95 latency, deadline misses, refresh load and budget hits. Report 95% paired bootstrap intervals; all-success/all-failure bootstrap intervals do not establish equivalence.

## Mechanism diagnostics

Use `--diagnostics` to measure relative change in refreshed suffix gains and the final actual-system importance-weight mass of samples that received refreshed gains. Diagnostics execute after the planning timer. Per-cycle trace adds feedforward, feedback, clipped applied control, nominal/actual gap, pole error and refresh influence. Very small importance weight would explain why refreshed gains do not affect the selected control. A paired rollout can also have zero actual/nominal tracking error while leaving the DDP reference radius, making gain refresh unnecessary for the feedback action.

No parameter is promoted to the production default solely because it reduces refresh frequency. The conclusion may be that V0 distance triggering is not useful on the tested cart-pole regimes. Existing unresolved racecheck findings continue to limit concurrency validation; this study does not relabel those reports as clean.

## Follow-up registered after the parameter screen

The initial screen completed on seeds 20-22. Raising cart cost to 200 produced 6/18 primary successes for each mode, versus 0/18 in the default setting. Neither mode gained a primary-success advantage in any screened configuration. Longer horizons restored nominal budget saturation and reduced pole-only success; early safety termination makes their cumulative-cost differences unsuitable for claiming improvement.

Follow-up on fresh development seeds 100-102: cart weights 100, 200, and 500, plus cart weight 200 combined separately with force 20 N, noise stddev 2.5, lambda 1, and three DDP iterations. Retain radius 4, horizon 50, and kicks 0.25, 0.5, 0.75, and 1 rad/s. This isolates whether improvements to the shared cart recovery setup create an adaptive opportunity near the recovery transition. Final selection still uses stock primary recovery first; any adaptive outcome difference is reported separately.

## Final configuration selection and trigger check

The follow-up on seeds 100-102 completed: cart weight 200 with force 20 N achieved 23/24 disturbed primary successes in both modes. All follow-up settings had equal paired primary success counts for stock and adaptive. This setting is selected as the second common configuration. Its nominal refresh demand at radius 4 was about 0.19/cycle, above the registered 0.1/cycle criterion. Before freezing evaluation, check radii 4, 4.5, 5 and 6 on seeds 200-204 with nominal and 1 rad/s matched/mismatched kicks; pick the smallest meeting the same nominal criterion.

The final evaluation grid is fixed now: nominal plus matched/mismatched kicks 0.25, 0.5, 0.75, 1, 1.25 and 1.5 rad/s; 30 paired seeds 5000-5029; 300 cycles. Evaluate (1) original shared parameters with calibrated radius 4 and (2) cart weight 200/force 20 N with the radius selected by the final nominal trigger check. Also run a nominal-only no-refresh control at radius 1e6 with the original shared parameters and the same evaluation seeds, to measure overhead that persists even when no policy is refreshed. No other settings or criteria are changed after the held-out run begins.

The final trigger check selected radius **4.5** for cart weight 200/force 20 N: nominal demand was 0.0067/cycle at 4.5 versus 0.207/cycle at 4; both had no budget hits. Final settings are now frozen: default/radius 4, cart200-force20/radius 4.5, and the default nominal-only radius-1e6 control. Evaluation seeds 5000-5029 begin after this selection.
