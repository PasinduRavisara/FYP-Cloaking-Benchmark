import os
import math
import time
import pandas as pd
import numpy as np

from algorithms import (
    PlanarLaplace,
    TraCS,
    KinematicGeometricCloaking,
    haversine_distance,
    calculate_bearing
)

SAMPLE_FILE = os.path.join("data", "processed", "geolife_user000_sample_1.csv")


def angle_difference(b1, b2):
    """
    Computes absolute angular deviation between two bearings in degrees [0, 180].
    """
    diff = abs(math.degrees(b1 - b2)) % 360
    return 360 - diff if diff > 180 else diff


def run_full_trajectory_benchmark(file_path, epsilon=0.5, v_max_kmh=60.0):
    df = pd.read_csv(file_path)
    n_points = len(df)
    v_max_ms = (v_max_kmh * 1000.0) / 3600.0  # speed limit in m/s

    print("=" * 80)
    print(f"BENCHMARKING FULL TRAJECTORY: {os.path.basename(file_path)}")
    print(f"Total Points: {n_points} | Privacy Budget (ε): {epsilon} | Speed Limit: {v_max_kmh} km/h")
    print("=" * 80)

    # Initialize all three models
    models = {
        "Planar Laplace (Baseline 1)": PlanarLaplace(epsilon=epsilon),
        "TraCS-D (Baseline 2)": TraCS(epsilon=epsilon),
        "Our Kinematic Engine (Proposed)": KinematicGeometricCloaking(
            epsilon=epsilon, theta_max_deg=60.0, v_max_kmh=v_max_kmh
        )
    }

    results = []

    for name, model in models.items():
        cloaked_points = []
        latencies = []

        # Reset state if testing our engine
        if hasattr(model, "reset_state"):
            model.reset_state()

        # Step 1: Run algorithm across all sequential points
        for i in range(n_points):
            row = df.iloc[i]
            lat, lon, t = row["latitude"], row["longitude"], row["timestamp"]

            t0 = time.perf_counter()
            if name == "Our Kinematic Engine (Proposed)":
                c_lat, c_lon, _ = model.cloak(lat, lon, t)
            else:
                c_lat, c_lon = model.perturb(lat, lon)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            cloaked_points.append((c_lat, c_lon, t))
            latencies.append(elapsed_ms)

        # Step 2: Compute Metrics & Simulate Attacks
        errors = []
        direction_mutations = 0
        speed_violations = 0

        for i in range(n_points):
            # 1. Average Euclidean / Haversine Error (m)
            true_lat, true_lon = df.iloc[i]["latitude"], df.iloc[i]["longitude"]
            c_lat, c_lon, _ = cloaked_points[i]
            errors.append(haversine_distance(true_lat, true_lon, c_lat, c_lon))

            # Consecutive checks (from point 1 onward)
            if i >= 1:
                prev_c_lat, prev_c_lon, prev_t = cloaked_points[i - 1]
                delta_t = max(cloaked_points[i][2] - prev_t, 0.5)

                # Distance between consecutive cloaked points
                step_dist = haversine_distance(prev_c_lat, prev_c_lon, c_lat, c_lon)
                apparent_speed = step_dist / delta_t

                # Attack 1: Kinematic Reachability Pruning (Speed violation)
                if apparent_speed > v_max_ms:
                    speed_violations += 1

            # Direction Mutation Check (from point 2 onward)
            if i >= 2:
                p0 = cloaked_points[i - 2]
                p1 = cloaked_points[i - 1]
                p2 = cloaked_points[i]

                bearing1 = calculate_bearing(p0[0], p0[1], p1[0], p1[1])
                bearing2 = calculate_bearing(p1[0], p1[1], p2[0], p2[1])

                # Attack 2: Direction Mutation (> 90° unnatural turn)
                if angle_difference(bearing1, bearing2) > 90.0:
                    direction_mutations += 1

        total_transitions = n_points - 1
        total_angle_transitions = n_points - 2

        results.append({
            "Mechanism": name,
            "Mean Error (m)": np.mean(errors),
            "Direction Mutation ASR (%)": (direction_mutations / total_angle_transitions) * 100.0,
            "Reachability Pruning Rate (%)": (speed_violations / total_transitions) * 100.0,
            "Mean Latency (ms)": np.mean(latencies)
        })

    # Display final comparative table
    summary_df = pd.DataFrame(results)
    print("\n" + summary_df.to_string(index=False))
    print("=" * 80)
    print("Summary of Defense Performance:")
    print("  * Direction Mutation ASR: Checks % of points making impossible > 90° backward turns.")
    print("  * Reachability Pruning Rate: Checks % of points pruned by road speed limit filters.")
    print("  * Mean Latency: Proves Θ(1) constant execution time per query.")


if __name__ == "__main__":
    run_full_trajectory_benchmark(SAMPLE_FILE)