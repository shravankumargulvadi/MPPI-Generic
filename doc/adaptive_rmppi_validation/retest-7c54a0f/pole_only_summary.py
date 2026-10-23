"""Derive a secondary pole angle/rate recovery summary from benchmark traces.

A pole is recovered if sqrt(w_theta * wrap(theta-pi)^2 + w_theta_dot * theta_dot^2)
falls below 0.2 by 2 seconds after the kick and stays below it through the remaining
recorded trace. This omits cart position and track-limit criteria; it never replaces
the benchmark's primary combined recovery result.
"""
import csv
import math
import sys
from collections import defaultdict

trace_path, output_path = sys.argv[1:3]
by_trial = defaultdict(list)
with open(trace_path, newline="") as f:
    for row in csv.DictReader(f):
        key = (row["scenario"], float(row["severity"]), row["mode"], int(row["seed"]))
        by_trial[key].append(row)

summary = defaultdict(list)
for (scenario, severity, mode, seed), rows in by_trial.items():
    rows.sort(key=lambda r: int(r["cycle"]))
    post = [r for r in rows if int(r["cycle"]) >= 50]
    error = []
    for row in post:
        dtheta = math.atan2(math.sin(float(row["theta"]) - math.pi),
                            math.cos(float(row["theta"]) - math.pi))
        error.append(math.sqrt(dtheta * dtheta + 0.1 * float(row["theta_dot"]) ** 2))
    recovery_s = -1.0
    for i, row in enumerate(post):
        elapsed = float(row["time_s"]) - 1.0
        if elapsed <= 2.0 and error[i] < 0.2 and all(e < 0.2 for e in error[i:]):
            recovery_s = (int(row["cycle"]) - 50) * 0.02
            break
    summary[(scenario, severity, mode)].append(recovery_s)

with open(output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["scenario", "severity", "mode", "trials", "pole_only_recoveries", "mean_recovery_s"])
    for (scenario, severity, mode), times in sorted(summary.items()):
        recovered = [t for t in times if t >= 0]
        writer.writerow([scenario, severity, mode, len(times), len(recovered),
                         f"{sum(recovered) / len(recovered):.6f}" if recovered else "NA"])
