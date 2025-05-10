import os
import pandas as pd
from glob import glob

RAW_DIR = "data/raw"
PROCESSED_PATH = "data/processed/jc_all_cleaned.csv"
os.makedirs("data/processed", exist_ok=True)

def load_and_clean_data():
    # Load all JC files
    csv_files = sorted(glob(os.path.join(RAW_DIR, "JC-*.csv")))
    print(f"📦 Found {len(csv_files)} raw JC files.")

    all_dfs = []

    for file in csv_files:
        print(f"🔄 Processing: {file}")
        try:
            df = pd.read_csv(file)

            # Strip milliseconds (e.g., .123) from time strings
            df["started_at"] = df["started_at"].astype(str).str.split(".").str[0]
            df["ended_at"] = df["ended_at"].astype(str).str.split(".").str[0]

            # Parse cleaned strings into datetime
            df["started_at"] = pd.to_datetime(df["started_at"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
            df["ended_at"] = pd.to_datetime(df["ended_at"], format="%Y-%m-%d %H:%M:%S", errors="coerce")

            # Drop rows with missing or invalid critical fields
            df.dropna(subset=["started_at", "ended_at", "start_station_id", "end_station_id"], inplace=True)

            # Add date parts for later feature engineering
            df["date"] = df["started_at"].dt.date
            df["hour"] = df["started_at"].dt.hour
            df["weekday"] = df["started_at"].dt.dayofweek

            all_dfs.append(df)
        except Exception as e:
            print(f"❌ Skipping {file} due to error: {e}")

    # Combine all cleaned data
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df.to_csv(PROCESSED_PATH, index=False)

    print(f"✅ Cleaned data saved to {PROCESSED_PATH}")
    print("📊 Year distribution:")
    print(full_df["started_at"].dt.year.value_counts().sort_index())

if __name__ == "__main__":
    load_and_clean_data()
