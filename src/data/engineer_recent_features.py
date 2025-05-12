import pandas as pd
import os

INPUT_PATH = "data/processed/jc_recent_cleaned.csv"
OUTPUT_PATH = "data/processed/jc_recent_hourly_features.csv"
os.makedirs("data/processed", exist_ok=True)

def engineer_features():
    print(f"📥 Loading cleaned data from {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH, parse_dates=["started_at"])

    # Extract hour and station ID
    df["hour"] = df["started_at"].dt.floor("H")
    df["start_station_id"] = df["start_station_id"].astype(str)

    # Aggregate: rides per hour per station
    hourly = (
        df.groupby(["start_station_id", "hour"])
        .size()
        .reset_index(name="rides")
        .sort_values(["start_station_id", "hour"])
    )

    # Add lag features
    for lag in range(1, 29):
        hourly[f"lag_{lag}"] = hourly.groupby("start_station_id")["rides"].shift(lag)

    # Add time-based features
    hourly["hour_of_day"] = hourly["hour"].dt.hour
    hourly["day_of_week"] = hourly["hour"].dt.dayofweek

    # Drop rows with incomplete lag values
    hourly.dropna(inplace=True)

    # Export
    hourly.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Engineered features saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    engineer_features()
