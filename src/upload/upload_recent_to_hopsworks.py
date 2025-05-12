import os
import pandas as pd
import hopsworks

FEATURE_GROUP_NAME = "citibike_features_dataset"
FEATURE_GROUP_VERSION = 1
INPUT_PATH = "data/processed/jc_recent_hourly_features.csv"

def upload_incremental():
    print("🔐 Logging in to Hopsworks...")
    project = hopsworks.login(
        project=os.environ["HOPSWORKS_PROJECT_NAME"],
        api_key_value=os.environ["HOPSWORKS_API_KEY"]
    )
    fs = project.get_feature_store()

    # Load engineered data
    df = pd.read_csv(INPUT_PATH, parse_dates=["hour"])
    print(f"📈 Loaded {len(df)} rows from {INPUT_PATH}")

    # Get existing feature group
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    existing_df = fg.read()
    latest_hour = existing_df["hour"].max()
    print(f"📅 Latest hour in Hopsworks: {latest_hour}")

    # Filter: Only new rows
    df_to_upload = df[df["hour"] > latest_hour].copy()
    df_to_upload.drop_duplicates(subset=["hour", "start_station_id"], keep="last", inplace=True)

    if df_to_upload.empty:
        print("🚫 No new data to upload. Exiting safely.")
        return

    print(f"⬆️ Uploading {len(df_to_upload)} new rows to Hopsworks...")
    fg.insert(df_to_upload, write_options={"wait_for_job": True, "write_mode": "append"})
    print("✅ Upload complete!")

if __name__ == "__main__":
    upload_incremental()
