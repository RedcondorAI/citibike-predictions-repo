import os
import pandas as pd
import hopsworks
from dotenv import load_dotenv

load_dotenv()

RESULTS_DIR = "data/metrics"
FUTURE_PREFIX = "citibike_future_predictions_dataset"

def upload_future_pca(project):
    fs = project.get_feature_store()

    for fname in os.listdir(RESULTS_DIR):
        if not fname.startswith("future_lgbm_pca_") or not fname.endswith(".csv"):
            continue

        try:
            parts = fname.replace(".csv", "").split("_")
            modeltype = parts[2]  # should be 'pca'
            station = parts[3].lower()
            fg_name = f"{FUTURE_PREFIX}_{modeltype}_{station}"

            df = pd.read_csv(os.path.join(RESULTS_DIR, fname), parse_dates=["hour"])
            fg = fs.get_or_create_feature_group(
                name=fg_name,
                version=1,
                primary_key=["hour"],
                event_time="hour"
            )
            fg.insert(df)
            print(f"✅ Uploaded: {fname} → {fg_name}")

        except Exception as e:
            print(f"❌ Failed to upload {fname}: {e}")

def main():
    project = hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT_NAME"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY")
    )
    upload_future_pca(project)

if __name__ == "__main__":
    main()
