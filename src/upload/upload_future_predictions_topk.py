import os
import pandas as pd
import hopsworks
from dotenv import load_dotenv

load_dotenv()

RESULTS_DIR = "data/metrics"
FUTURE_PREFIX = "citibike_future_predictions_dataset"
TOP_STATIONS = ["JC115", "HB102", "HB103"]

def upload_future_topk(project):
    fs = project.get_feature_store()

    for station in TOP_STATIONS:
        fname = f"future_lgbm_topk_{station}.csv"
        path = os.path.join(RESULTS_DIR, fname)

        if not os.path.exists(path):
            print(f"❌ Missing: {fname}")
            continue

        try:
            df = pd.read_csv(path, parse_dates=["hour"])
            fg = fs.get_or_create_feature_group(
                name=f"{FUTURE_PREFIX}_topk_{station.lower()}",
                version=1,
                primary_key=["hour"],
                event_time="hour"
            )
            fg.insert(df)
            print(f"✅ Uploaded: {fname} → {FUTURE_PREFIX}_topk_{station.lower()}")
        except Exception as e:
            print(f"❌ Failed to upload {fname}: {e}")

def main():
    project = hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT_NAME"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY")
    )
    upload_future_topk(project)

if __name__ == "__main__":
    main()
