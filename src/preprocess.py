import os
import glob
from datetime import datetime
import pandas as pd

RAW_GEOLIFE_DIR = os.path.join("data", "raw", "geolife")
RAW_TDRIVE_DIR = os.path.join("data", "raw", "tdrive")
PROCESSED_DIR = os.path.join("data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)


def parse_geolife_plt(file_path):
    """
    Parses a single GeoLife .plt file.
    Skips the 6-line header. Extracts [timestamp, latitude, longitude].
    """
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()[6:]

    for line in lines:
        parts = line.strip().split(",")
        if len(parts) >= 7:
            lat = float(parts[0])
            lon = float(parts[1])
            dt_str = f"{parts[5]} {parts[6]}"
            dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            records.append({
                "timestamp": dt.timestamp(),
                "datetime": dt_str,
                "latitude": lat,
                "longitude": lon
            })
    return pd.DataFrame(records)


def extract_continuous_tdrive_trips(file_path, max_gap_seconds=600.0, min_trip_len=150):
    """
    Extracts continuous driving trips from a taxi's log by splitting
    only when the vehicle is parked or silent for more than 10 minutes (600s).
    """
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 4:
                dt_str = parts[1]
                lon = float(parts[2])
                lat = float(parts[3])
                dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                records.append({
                    "timestamp": dt.timestamp(),
                    "datetime": dt_str,
                    "latitude": lat,
                    "longitude": lon
                })
    df = pd.DataFrame(records)
    if df.empty:
        return []

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["time_gap"] = df["timestamp"].diff().fillna(0)

    # Split into separate trips if gap exceeds 10 minutes (600 seconds)
    df["trip_id"] = (df["time_gap"] > max_gap_seconds).cumsum()

    continuous_trips = []
    for _, trip_df in df.groupby("trip_id"):
        if len(trip_df) >= min_trip_len:
            clean_trip = trip_df[["timestamp", "datetime", "latitude", "longitude"]].reset_index(drop=True)
            continuous_trips.append(clean_trip)

    return continuous_trips


def extract_benchmark_samples():
    print("Extracting GeoLife benchmark trajectories...")
    geolife_saved = 0
    for user_id in ["000", "001", "002", "003", "004"]:
        traj_pattern = os.path.join(RAW_GEOLIFE_DIR, user_id, "Trajectory", "*.plt")
        files = glob.glob(traj_pattern)
        for filepath in files:
            df = parse_geolife_plt(filepath)
            if 150 <= len(df) <= 1000:
                out_name = f"geolife_user{user_id}_sample_{geolife_saved + 1}.csv"
                df.to_csv(os.path.join(PROCESSED_DIR, out_name), index=False)
                geolife_saved += 1
                if geolife_saved >= 5:
                    break
        if geolife_saved >= 5:
            break
    print(f" Saved {geolife_saved} GeoLife routes.")

    print("\nExtracting continuous T-Drive taxi trips (10-minute traffic-tolerant window)...")
    tdrive_saved = 0
    for taxi_file in sorted(os.listdir(RAW_TDRIVE_DIR)):
        if taxi_file.endswith(".txt"):
            full_path = os.path.join(RAW_TDRIVE_DIR, taxi_file)
            trips = extract_continuous_tdrive_trips(full_path, max_gap_seconds=600.0, min_trip_len=150)
            for trip_df in trips:
                taxi_id = taxi_file.split(".")[0]
                out_name = f"tdrive_taxi{taxi_id}_sample_{tdrive_saved + 1}.csv"
                trip_df.to_csv(os.path.join(PROCESSED_DIR, out_name), index=False)
                tdrive_saved += 1
                if tdrive_saved >= 5:
                    break
            if tdrive_saved >= 5:
                break
    print(f" Saved {tdrive_saved} true continuous T-Drive taxi trips.")


if __name__ == "__main__":
    extract_benchmark_samples()