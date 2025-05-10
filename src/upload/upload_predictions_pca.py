import os
import shutil
import joblib
import pandas as pd
import hopsworks
from hsml.model_schema import ModelSchema
from hsml.schema import Schema
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths and Constants
RESULTS_DIR = "data/metrics"
MODELS_DIR = "trained_models"
TOP_STATIONS = ["JC115", "HB102", "HB103"]
PCA_MODEL_NAME = "citibike_pca"
PCA_METRICS_FILE = "lgbm_pca_mae_summary.csv"


def upload_pca_models(project):
    registry = project.get_model_registry()
    mae_path = os.path.join(RESULTS_DIR, PCA_METRICS_FILE)
    mae_df = pd.read_csv(mae_path) if os.path.exists(mae_path) else pd.DataFrame()

    for station_id in TOP_STATIONS:
        model_dir = os.path.join(MODELS_DIR, f"pca_model_{station_id}")
        model_path = os.path.join(model_dir, "lgbm_model.pkl")
        pca_path = os.path.join(model_dir, "pca_transformer.pkl")

        if not os.path.exists(model_path) or not os.path.exists(pca_path):
            print(f"❌ Missing files for {station_id}")
            continue

        tmp_dir = f"tmp_pca_model_{station_id}"
        os.makedirs(tmp_dir, exist_ok=True)
        shutil.copy(model_path, os.path.join(tmp_dir, "model.pkl"))
        shutil.copy(pca_path, os.path.join(tmp_dir, "pca_transformer.pkl"))

        mae = 0
        explained_variance = 0
        if not mae_df.empty:
            try:
                row = mae_df.loc[mae_df["station_id"] == station_id]
                mae = float(row["mae"].values[0])
                explained_variance = float(row["explained_variance"].values[0])
            except Exception:
                pass

        # Dummy schema for registration
        X_dummy = pd.DataFrame({f"lag_{i}": [0.0] for i in range(1, 29)})
        y_dummy = pd.Series([0.0])

        model = registry.python.create_model(
            name=f"{PCA_MODEL_NAME}_{station_id}",
            metrics={"mae": mae, "explained_variance": explained_variance},
            model_schema=ModelSchema(Schema(X_dummy), Schema(y_dummy)),
            description=f"PCA + LightGBM model for {station_id}"
        )
        model.save(tmp_dir)
        print(f"✅ Uploaded PCA model for {station_id}")

        # Clean up temporary directory
        shutil.rmtree(tmp_dir)


def main():
    project = hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT_NAME"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY")
    )
    upload_pca_models(project)


if __name__ == "__main__":
    main()
