import pandas as pd
import os

INPUT_PATH = "data/processed/jc_all_cleaned.csv"
OUTPUT_PATH = "data/processed/jc_hourly_features.csv"
os.makedirs("data/processed", exist_ok=True)

def engineer_features():
    df = pd.read_csv(INPUT_PATH)
    df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")


    # Extract time parts
    df["hour"] = df["started_at"].dt.floor("H")  # floor to hour
    df["start_station_id"] = df["start_station_id"].astype(str)

    # Aggregate: ride count per hour per start station
    hourly = (
        df.groupby(["start_station_id", "hour"])
        .size()
        .reset_index(name="rides")
        .sort_values(["start_station_id", "hour"])
    )

    # Add lag features (past 1 to 28 hours)
    for lag in range(1, 29):
        hourly[f"lag_{lag}"] = (
            hourly.groupby("start_station_id")["rides"].shift(lag)
        )

    # Add time-based features
    hourly["hour_of_day"] = hourly["hour"].dt.hour
    hourly["day_of_week"] = hourly["hour"].dt.dayofweek

    # Drop rows with NA in lag columns (first 28 rows per station)
    
    hourly.dropna(inplace=True)

    hourly.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved engineered features to {OUTPUT_PATH}")

if __name__ == "__main__":
    engineer_features()
