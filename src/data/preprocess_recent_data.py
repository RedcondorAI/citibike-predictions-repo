import os
import pandas as pd
from glob import glob

RAW_DIR = "data/raw"
OUTPUT_PATH = "data/processed/jc_recent_cleaned.csv"
os.makedirs("data/processed", exist_ok=True)

def get_most_recent_files(n=2):
    """Return the n most recently modified JC CSV files."""
    files = sorted(glob(os.path.join(RAW_DIR, "JC-*.csv")), key=os.path.getmtime, reverse=True)
    return files[:n]

def preprocess(files):
    all_dfs = []

    for file in files:
        print(f"🔄 Processing: {file}")
        try:
            df = pd.read_csv(file)

            # Remove milliseconds from timestamps
            df["started_at"] = df["started_at"].astype(str).str.split(".").str[0]
            df["ended_at"] = df["ended_at"].astype(str).str.split(".").str[0]

            # Convert to datetime
            df["started_at"] = pd.to_datetime(df["started_at"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
            df["ended_at"] = pd.to_datetime(df["ended_at"], format="%Y-%m-%d %H:%M:%S", errors="coerce")

            # Drop nulls in critical columns
            df.dropna(subset=["started_at", "ended_at", "start_station_id", "end_station_id"], inplace=True)

            # Optional: Keep only useful columns
            df["date"] = df["started_at"].dt.date
            df["hour"] = df["started_at"].dt.hour
            df["weekday"] = df["started_at"].dt.dayofweek

            all_dfs.append(df)
        except Exception as e:
            print(f"❌ Error processing {file}: {e}")

    # Combine and export
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(OUTPUT_PATH, index=False)
        print(f"✅ Cleaned data saved to {OUTPUT_PATH}")
    else:
        print("⚠️ No data processed.")

if __name__ == "__main__":
    recent_files = get_most_recent_files(n=2)
    preprocess(recent_files)
