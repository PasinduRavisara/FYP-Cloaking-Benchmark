import os
import glob
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

PROCESSED_DIR = os.path.join("data", "processed")
FIGURES_DIR = "figures"
RESULTS_DIR = "results"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Consistent color scheme across all plots
COLORS = {
    "True Route": "#1E293B",        # Deep Slate Gray
    "Planar Laplace": "#E11D48",    # Coral Red
    "TraCS-D": "#D97706",           # Amber Orange
    "Cloaking Engine": "#059669"    # Emerald Green
}


def angle_difference(b1, b2):
    """Computes absolute angular deviation between two bearings in degrees [0, 180]."""
    diff = abs(math.degrees(b1 - b2)) % 360
    return 360 - diff if diff > 180 else diff


def compute_dtw_distance(true_coords, cloaked_coords, window=15):
    """
    Computes Sakoe-Chiba banded Dynamic Time Warping (DTW) distance (normalized per point in meters).
    Measures overall trajectory shape similarity between True Route and Cloaked Route.
    """
    n = len(true_coords)
    m = len(cloaked_coords)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0

    w = max(window, abs(n - m))
    for i in range(1, n + 1):
        start_j = max(1, i - w)
        end_j = min(m, i + w) + 1
        t_lat, t_lon = true_coords[i - 1][0], true_coords[i - 1][1]
        for j in range(start_j, end_j):
            c_lat, c_lon = cloaked_coords[j - 1][0], cloaked_coords[j - 1][1]
            cost = haversine_distance(t_lat, t_lon, c_lat, c_lon)
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    return dtw[n, m] / n


def evaluate_single_trajectory(df, dataset_type, epsilon=0.5):
    """
    Runs all 3 mechanisms on a single trajectory DataFrame and computes all 5 metrics:
    Mean Error (AE), DTW Distance, Direction Mutation ASR, Reachability Pruning Rate, and Latency.
    """
    n_points = len(df)
    v_max_kmh = 60.0 if dataset_type == "GeoLife" else 90.0
    v_max_ms = (v_max_kmh * 1000.0) / 3600.0

    models = {
        "Planar Laplace": PlanarLaplace(epsilon=epsilon),
        "TraCS-D": TraCS(epsilon=epsilon),
        "Cloaking Engine": KinematicGeometricCloaking(
            epsilon=epsilon, theta_max_deg=60.0, v_max_kmh=v_max_kmh
        )
    }

    true_coords = df[["latitude", "longitude", "timestamp"]].values
    file_metrics = {}
    cloaked_traces = {"True Route": true_coords}

    for mech_name, model in models.items():
        if hasattr(model, "reset_state"):
            model.reset_state()

        cloaked_points = []
        latencies = []

        for i in range(n_points):
            lat, lon, t = true_coords[i][0], true_coords[i][1], true_coords[i][2]
            t0 = time.perf_counter()
            if mech_name == "Cloaking Engine":
                c_lat, c_lon, _ = model.cloak(lat, lon, t)
            else:
                c_lat, c_lon = model.perturb(lat, lon)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            cloaked_points.append((c_lat, c_lon, t))

        cloaked_arr = np.array(cloaked_points)
        cloaked_traces[mech_name] = cloaked_arr

        # 1. Average Euclidean / Haversine Error (AE in meters)
        errors = [
            haversine_distance(true_coords[i][0], true_coords[i][1], cloaked_arr[i][0], cloaked_arr[i][1])
            for i in range(n_points)
        ]

        # 2. Dynamic Time Warping (DTW in meters per point)
        dtw_val = compute_dtw_distance(true_coords, cloaked_arr)

        # 3. Kinematic Reachability Pruning Rate (%)
        speed_violations = 0
        for i in range(1, n_points):
            dist = haversine_distance(cloaked_arr[i-1][0], cloaked_arr[i-1][1], cloaked_arr[i][0], cloaked_arr[i][1])
            dt = max(cloaked_arr[i][2] - cloaked_arr[i-1][2], 0.5)
            if (dist / dt) > v_max_ms:
                speed_violations += 1

        # 4. Direction Mutation Attack Success Rate (%)
        mutations = 0
        for i in range(2, n_points):
            b1 = calculate_bearing(cloaked_arr[i-2][0], cloaked_arr[i-2][1], cloaked_arr[i-1][0], cloaked_arr[i-1][1])
            b2 = calculate_bearing(cloaked_arr[i-1][0], cloaked_arr[i-1][1], cloaked_arr[i][0], cloaked_arr[i][1])
            if angle_difference(b1, b2) > 90.0:
                mutations += 1

        file_metrics[mech_name] = {
            "Mean Error (m)": np.mean(errors),
            "DTW Distance (m)": dtw_val,
            "Direction Mutation ASR (%)": (mutations / (n_points - 2)) * 100.0,
            "Reachability Pruning Rate (%)": (speed_violations / (n_points - 1)) * 100.0,
            "Mean Latency (ms)": np.mean(latencies)
        }

    return file_metrics, cloaked_traces


def plot_spatial_trajectory_map(traces, title_label, out_filename):
    """Generates a 3-panel 2D trajectory comparison map."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.15)

    true_lats = traces["True Route"][:, 0]
    true_lons = traces["True Route"][:, 1]

    for idx, mech_name in enumerate(["Planar Laplace", "TraCS-D", "Cloaking Engine"]):
        ax = axes[idx]
        ax.plot(true_lons, true_lats, color="#94A3B8", linewidth=2.0, linestyle="--", label="True GPS Route", zorder=1)
        ax.scatter(true_lons[0], true_lats[0], color="#1E293B", marker="o", s=70, label="Start Point", zorder=4)
        ax.scatter(true_lons[-1], true_lats[-1], color="#1E293B", marker="X", s=90, label="End Point", zorder=4)

        c_lats = traces[mech_name][:, 0]
        c_lons = traces[mech_name][:, 1]

        ax.plot(c_lons, c_lats, color=COLORS[mech_name], linewidth=1.4, alpha=0.85, label=f"{mech_name} Output", zorder=2)
        ax.scatter(c_lons, c_lats, color=COLORS[mech_name], s=12, alpha=0.5, zorder=3)

        ax.set_title(f"True Route vs. {mech_name}", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Longitude (°E)", fontsize=11, labelpad=8)
        if idx == 0:
            ax.set_ylabel("Latitude (°N)", fontsize=11, labelpad=8)

        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best", framealpha=0.92, fontsize=9.5)

    fig.suptitle(title_label, fontsize=15, fontweight="bold", y=1.00)
    out_path = os.path.join(FIGURES_DIR, out_filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Updated Figure] {out_path}")


def plot_single_route_barchart(metrics, n_points, epsilon, out_filename):
    """Updates benchmark_metrics_barchart.png on every run."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    plt.subplots_adjust(wspace=0.3)

    mech_names = ["Planar Laplace", "TraCS-D", "Cloaking Engine"]
    bar_colors = [COLORS[m] for m in mech_names]

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

    fig.suptitle(f"Empirical Evaluation Across {n_points} Consecutive Trajectory Points (ε = {epsilon})", fontsize=15, fontweight="bold", y=1.03)
    out_path = os.path.join(FIGURES_DIR, out_filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Updated Figure] {out_path}")


def plot_multi_dataset_bars(summary_df):
    """Generates a 2x2 grouped bar chart comparing GeoLife and T-Drive across all 3 mechanisms."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    datasets = ["GeoLife", "T-Drive"]
    mechanisms = ["Planar Laplace", "TraCS-D", "Cloaking Engine"]
    x = np.arange(len(datasets))
    width = 0.25

    configs = [
        ("Reachability Pruning Rate (%)", axes[0, 0], "Adversarial Speed Pruning Rate (%)\n[Lower = Stronger Privacy Defense]", "{:.1f}%", 115),
        ("Direction Mutation ASR (%)", axes[0, 1], "Direction Mutation Attack ASR (%)\n[Lower = Stronger Privacy Defense]", "{:.1f}%", 95),
        ("Mean Error (m)", axes[1, 0], "Average Spatial Error (AE in meters)\n[Lower = Better Route Utility]", "{:.1f} m", 480),
        ("DTW Distance (m)", axes[1, 1], "Dynamic Time Warping (DTW in meters)\n[Lower = Better Trajectory Shape]", "{:.1f} m", 420)
    ]

    for metric_col, ax, title_str, fmt_str, y_max in configs:
        for i, mech in enumerate(mechanisms):
            vals = [
                summary_df[(summary_df["Dataset"] == ds) & (summary_df["Mechanism"] == mech)][metric_col].values[0]
                for ds in datasets
            ]
            offset = (i - 1) * width
            bars = ax.bar(x + offset, vals, width, label=mech, color=COLORS[mech], edgecolor="#0F172A", linewidth=1.0, zorder=3)

            for bar in bars:
                h = bar.get_height()
                ax.annotate(
                    fmt_str.format(h),
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9.5,
                    fontweight="bold"
                )

        ax.set_title(title_str, fontsize=12.5, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(["Microsoft GeoLife (5 Trips)", "Beijing T-Drive (5 Taxi Trips)"], fontsize=11, fontweight="bold")
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", linestyle=":", alpha=0.7, zorder=0)
        ax.legend(framealpha=0.95, fontsize=10)

    fig.suptitle("Multi-Dataset Empirical Evaluation: GeoLife vs. T-Drive (ε = 0.5)", fontsize=15, fontweight="bold", y=0.98)
    out_path = os.path.join(FIGURES_DIR, "multi_dataset_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Updated Figure] {out_path}")


def run_epsilon_sensitivity(sample_df):
    """Sweeps across ε ∈ [0.1, 0.5, 1.0, 2.0, 4.0] and generates line charts."""
    epsilons = [0.1, 0.5, 1.0, 2.0, 4.0]
    mechanisms = ["Planar Laplace", "TraCS-D", "Cloaking Engine"]
    sweep_records = []

    for eps in epsilons:
        metrics, _ = evaluate_single_trajectory(sample_df, "GeoLife", epsilon=eps)
        for mech in mechanisms:
            sweep_records.append({
                "Epsilon": eps,
                "Mechanism": mech,
                **metrics[mech]
            })

    sweep_df = pd.DataFrame(sweep_records)
    sweep_df.to_csv(os.path.join(RESULTS_DIR, "epsilon_sensitivity_results.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    plt.subplots_adjust(wspace=0.25)

    line_configs = [
        ("Reachability Pruning Rate (%)", axes[0], "Adversarial Pruning Rate vs. ε\n[Lower = Stronger Defense]", (0, 105)),
        ("Direction Mutation ASR (%)", axes[1], "Direction Mutation ASR vs. ε\n[Lower = Stronger Defense]", (0, 90)),
        ("Mean Error (m)", axes[2], "Mean Spatial Error (m) vs. ε\n[Log Scale — Lower = Higher Utility]", None)
    ]

    for metric_col, ax, title_str, y_lim in line_configs:
        for mech in mechanisms:
            sub = sweep_df[sweep_df["Mechanism"] == mech]
            ax.plot(
                sub["Epsilon"], sub[metric_col],
                marker="o", markersize=8, linewidth=2.5,
                color=COLORS[mech], label=mech
            )
        ax.set_title(title_str, fontsize=12.5, fontweight="bold", pad=10)
        ax.set_xlabel("Privacy Budget (ε) [Smaller = Stricter Privacy]", fontsize=11, labelpad=8)
        ax.set_xticks(epsilons)
        if y_lim:
            ax.set_ylim(y_lim)
        else:
            ax.set_yscale("log")
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.legend(framealpha=0.95, fontsize=10)

    fig.suptitle("Privacy Budget (ε) Sensitivity Analysis Across 908 Consecutive GPS Updates", fontsize=15, fontweight="bold", y=1.02)
    out_path = os.path.join(FIGURES_DIR, "epsilon_sensitivity_analysis.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Updated Figure] {out_path}")


def main():
    # Dynamic random state on every run (no fixed seed) so charts & metrics update live!
    csv_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*.csv")))

    print("=" * 90)
    print(f"RUNNING LIVE EMPIRICAL BENCHMARK SUITE ACROSS {len(csv_files)} TRAJECTORIES")
    print("=" * 90)

    all_rows = []
    geolife_map_plotted = False
    tdrive_map_plotted = False

    for filepath in csv_files:
        fname = os.path.basename(filepath)
        dataset_type = "GeoLife" if fname.startswith("geolife") else "T-Drive"
        df = pd.read_csv(filepath)

        print(f"-> Processing {fname} ({dataset_type}, {len(df)} points)...")
        metrics, traces = evaluate_single_trajectory(df, dataset_type, epsilon=0.5)

        for mech_name, m_vals in metrics.items():
            all_rows.append({
                "File": fname,
                "Dataset": dataset_type,
                "Points": len(df),
                "Mechanism": mech_name,
                **m_vals
            })

        # Update GeoLife spatial map & single-route barchart on the first GeoLife file
        if dataset_type == "GeoLife" and not geolife_map_plotted:
            plot_spatial_trajectory_map(
                traces,
                "Continuous Spatial Trajectory Obfuscation Comparison (Microsoft GeoLife)",
                "trajectory_comparison_map.png"
            )
            plot_single_route_barchart(
                metrics, len(df), 0.5, "benchmark_metrics_barchart.png"
            )
            geolife_map_plotted = True

        # Update T-Drive spatial comparison map on the first T-Drive route
        if dataset_type == "T-Drive" and not tdrive_map_plotted:
            plot_spatial_trajectory_map(
                traces,
                f"Continuous Vehicular Trajectory Obfuscation Comparison (Beijing T-Drive: {fname})",
                "tdrive_trajectory_comparison_map.png"
            )
            tdrive_map_plotted = True

    full_df = pd.DataFrame(all_rows)
    full_csv_path = os.path.join(RESULTS_DIR, "full_benchmark_results.csv")
    full_df.to_csv(full_csv_path, index=False)
    print(f"\n  [Updated Raw CSV] {full_csv_path}")

    # Group by Dataset and Mechanism to compute Mean across all 10 files
    summary_df = full_df.groupby(["Dataset", "Mechanism"], as_index=False)[
        ["Mean Error (m)", "DTW Distance (m)", "Direction Mutation ASR (%)", "Reachability Pruning Rate (%)", "Mean Latency (ms)"]
    ].mean()

    summary_csv_path = os.path.join(RESULTS_DIR, "summary_by_dataset.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    print("\n" + "=" * 90)
    print("AGGREGATE BENCHMARK SUMMARY (AVERAGED ACROSS ALL GEOLIFE & T-DRIVE ROUTES)")
    print("=" * 90)
    print(summary_df.to_string(index=False))
    print("=" * 90)

    # Generate multi-dataset bar chart
    print("\nUpdating Multi-Dataset Comparison Bar Chart...")
    plot_multi_dataset_bars(summary_df)

    # Run Epsilon Sensitivity Sweep on GeoLife Sample 1
    print("\nRunning Privacy Budget (ε) Sensitivity Sweep [ε = 0.1, 0.5, 1.0, 2.0, 4.0]...")
    geolife_1_df = pd.read_csv(os.path.join(PROCESSED_DIR, "geolife_user000_sample_1.csv"))
    run_epsilon_sensitivity(geolife_1_df)

    print("\nALL 5 FIGURES & 3 CSV TABLES UPDATED LIVE!")


if __name__ == "__main__":
    main()