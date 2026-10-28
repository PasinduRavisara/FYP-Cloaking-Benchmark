import os
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from algorithms import (
    PlanarLaplace,
    TraCS,
    KinematicGeometricCloaking,
    haversine_distance,
    calculate_bearing
)

SAMPLE_FILE = os.path.join("data", "processed", "geolife_user000_sample_1.csv")
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def angle_difference(b1, b2):
    diff = abs(math.degrees(b1 - b2)) % 360
    return 360 - diff if diff > 180 else diff


def run_and_generate_visualizations():
    print(f"Loading trajectory data: {SAMPLE_FILE} ...")
    df = pd.read_csv(SAMPLE_FILE)
    n_points = len(df)
    v_max_kmh = 60.0
    v_max_ms = (v_max_kmh * 1000.0) / 3600.0
    epsilon = 0.5

    # Color palette
    colors = {
        "True Route": "#1E293B",        # Deep Slate Gray
        "Planar Laplace": "#E11D48",    # Coral Red
        "TraCS-D": "#D97706",           # Amber Orange
        "Cloaking Engine": "#059669"    # Emerald Green
    }

    models = {
        "Planar Laplace": PlanarLaplace(epsilon=epsilon),
        "TraCS-D": TraCS(epsilon=epsilon),
        "Cloaking Engine": KinematicGeometricCloaking(
            epsilon=epsilon, theta_max_deg=60.0, v_max_kmh=v_max_kmh
        )
    }

    trajectories = {"True Route": df[["latitude", "longitude", "timestamp"]].values}
    metrics = {}

    for name, model in models.items():
        if hasattr(model, "reset_state"):
            model.reset_state()

        cloaked_points = []
        latencies = []

        for i in range(n_points):
            row = df.iloc[i]
            lat, lon, t = row["latitude"], row["longitude"], row["timestamp"]

            t0 = time.perf_counter()
            if name == "Cloaking Engine":
                c_lat, c_lon, _ = model.cloak(lat, lon, t)
            else:
                c_lat, c_lon = model.perturb(lat, lon)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            cloaked_points.append((c_lat, c_lon, t))

        trajectories[name] = np.array(cloaked_points)

        # Compute benchmark metrics
        errors = [
            haversine_distance(df.iloc[i]["latitude"], df.iloc[i]["longitude"], cloaked_points[i][0], cloaked_points[i][1])
            for i in range(n_points)
        ]

        speed_violations = 0
        for i in range(1, n_points):
            dist = haversine_distance(cloaked_points[i-1][0], cloaked_points[i-1][1], cloaked_points[i][0], cloaked_points[i][1])
            dt = max(cloaked_points[i][2] - cloaked_points[i-1][2], 0.5)
            if (dist / dt) > v_max_ms:
                speed_violations += 1

        mutations = 0
        for i in range(2, n_points):
            b1 = calculate_bearing(cloaked_points[i-2][0], cloaked_points[i-2][1], cloaked_points[i-1][0], cloaked_points[i-1][1])
            b2 = calculate_bearing(cloaked_points[i-1][0], cloaked_points[i-1][1], cloaked_points[i][0], cloaked_points[i][1])
            if angle_difference(b1, b2) > 90.0:
                mutations += 1

        metrics[name] = {
            "Mean Error (m)": np.mean(errors),
            "Direction Mutation ASR (%)": (mutations / (n_points - 2)) * 100.0,
            "Reachability Pruning Rate (%)": (speed_violations / (n_points - 1)) * 100.0,
            "Mean Latency (ms)": np.mean(latencies)
        }

    # =========================================================================
    # FIGURE 1: 2D SPATIAL TRAJECTORY MAP (4-Panel Side-by-Side Comparison)
    # =========================================================================
    print("Generating Figure 1: trajectory_comparison_map.png ...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.15)

    comparison_pairs = [
        ("Planar Laplace", axes[0]),
        ("TraCS-D", axes[1]),
        ("Cloaking Engine", axes[2])
    ]

    true_lats = trajectories["True Route"][:, 0]
    true_lons = trajectories["True Route"][:, 1]

    for name, ax in comparison_pairs:
        # Plot true route in background
        ax.plot(true_lons, true_lats, color="#94A3B8", linewidth=2.0, linestyle="--", label="True GPS Trajectory", zorder=1)
        ax.scatter(true_lons[0], true_lats[0], color="#1E293B", marker="o", s=70, label="Start Point", zorder=4)
        ax.scatter(true_lons[-1], true_lats[-1], color="#1E293B", marker="X", s=90, label="End Point", zorder=4)

        # Plot cloaked trajectory
        c_lats = trajectories[name][:, 0]
        c_lons = trajectories[name][:, 1]

        ax.plot(c_lons, c_lats, color=colors[name], linewidth=1.4, alpha=0.85, label=f"{name} Output", zorder=2)
        ax.scatter(c_lons, c_lats, color=colors[name], s=12, alpha=0.5, zorder=3)

        ax.set_title(f"True Route vs. {name}", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Longitude (°E)", fontsize=11, labelpad=8)
        if ax == axes[0]:
            ax.set_ylabel("Latitude (°N)", fontsize=11, labelpad=8)

        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower left", framealpha=0.92, fontsize=9.5)

    fig.suptitle("Continuous Spatial Trajectory Obfuscation Comparison (Microsoft GeoLife)", fontsize=15, fontweight="bold", y=1.00)
    map_filepath = os.path.join(OUTPUT_DIR, "trajectory_comparison_map.png")
    plt.savefig(map_filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" Saved: {map_filepath}")

    # =========================================================================
    # FIGURE 2: BENCHMARK METRICS BARCHART (4-Panel Publication Quality)
    # =========================================================================
    print("Generating Figure 2: benchmark_metrics_barchart.png ...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    plt.subplots_adjust(wspace=0.3)

    mech_names = list(models.keys())
    bar_colors = [colors[m] for m in mech_names]

    plot_configs = [
        ("Reachability Pruning Rate (%)", axes[0], "Adversarial Pruning Rate (%)\n[Lower = Better Defense]", "{:.1f}%", 110),
        ("Direction Mutation ASR (%)", axes[1], "Direction Mutation ASR (%)\n[Lower = Better Defense]", "{:.1f}%", 100),
        ("Mean Error (m)", axes[2], "Mean Spatial Error (m)\n[Lower = Higher Navigation Utility]", "{:.1f} m", max([metrics[m]["Mean Error (m)"] for m in mech_names]) * 1.25),
        ("Mean Latency (ms)", axes[3], "Computation Latency (ms)\n[Target < 5.0 ms]", "{:.4f} ms", 0.025)
    ]

    for metric_key, ax, title_str, fmt_str, y_lim in plot_configs:
        values = [metrics[m][metric_key] for m in mech_names]
        bars = ax.bar(mech_names, values, color=bar_colors, width=0.55, edgecolor="#0F172A", linewidth=1.1, zorder=3)

        ax.set_title(title_str, fontsize=11.5, fontweight="bold", pad=10)
        ax.set_ylim(0, y_lim)
        ax.grid(axis="y", linestyle=":", alpha=0.7, zorder=0)
        ax.tick_params(axis="x", rotation=12, labelsize=10)

        # Draw red dashed target line on latency plot
        if "Latency" in metric_key:
            ax.axhline(5.0, color="#DC2626", linestyle="--", linewidth=1.2, label="5 ms Bound", zorder=4)

        # Label bars with exact values
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                fmt_str.format(h),
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold"
            )

    fig.suptitle("Empirical Evaluation Across 908 Consecutive Trajectory Points (ε = 0.5)", fontsize=15, fontweight="bold", y=1.03)
    metrics_filepath = os.path.join(OUTPUT_DIR, "benchmark_metrics_barchart.png")
    plt.savefig(metrics_filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" Saved: {metrics_filepath}")

    print("\nAll visualizations successfully generated in 'figures/' directory!")


if __name__ == "__main__":
    run_and_generate_visualizations()