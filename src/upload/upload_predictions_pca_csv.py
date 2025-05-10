import os
import joblib
import pandas as pd
import hopsworks
from hsml.model_schema import ModelSchema
from hsml.schema import Schema
from dotenv import load_dotenv
import shutil

# Load environment variables
load_dotenv()

# Paths and Constants
RESULTS_DIR = "data/metrics"
MODELS_DIR = "trained_models"
TOP_STATIONS = ["JC115", "HB102", "HB103"]
PREDICTION_STORE_NAME = "citibike_predictions_dataset"
PCA_METRICS_FILE = os.path.join(RESULTS_DIR, "lgbm_pca_mae_summary.csv")

def upload_pca_predictions(project):
    fs = project.get_feature_store()
    for station_id in TOP_STATIONS:
        pred_path = os.path.join(RESULTS_DIR, f"predictions_lgbm_pca_{station_id}.csv")
        if not os.path.exists(pred_path):
            print(f"❌ Missing PCA prediction CSV for {station_id}")
            continue

        df = pd.read_csv(pred_path, parse_dates=["hour"])
        fg = fs.get_or_create_feature_group(
            name=f"{PREDICTION_STORE_NAME}_pca_{station_id.lower()}",
            version=1,
            primary_key=["hour"],
            event_time="hour"
        )
        fg.insert(df)
        print(f"✅ Uploaded PCA predictions for {station_id}")

def main():
    project = hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT_NAME"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY")
    )
    upload_pca_predictions(project)

if __name__ == "__main__":
    main()
